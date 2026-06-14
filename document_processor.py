import os
import logging
import rich.logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

os.environ["ANONYMIZED_TELEMETRY"]= "False"

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("RAG")

class DocumentProcessor:
    def __init__(self, doc_path: str = "knowledge_base", collection_name: str = "phoneguy_brain"):
        self.doc_path = os.path.abspath(doc_path)
        
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path, exist_ok=True)
        
        if not os.getenv("NVIDIA_API_KEY"):
            logger.warning("⚠️ No NVIDIA_API_KEY found. RAG will be disabled.")
            self.vector_store = None
            return

        try:
            self.embeddings = NVIDIAEmbeddings(
                model="nvidia/llama-nemotron-embed-vl-1b-v2",
                base_url=os.getenv("NVIDIA_API_BASE") or "https://integrate.api.nvidia.com/v1",
                nvidia_api_key=os.getenv("NVIDIA_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to init Embeddings: {e}")
            self.vector_store = None
            return

        self.persist_directory = os.path.abspath(f"chroma_db/{collection_name}")
        
        try:
            # Инициализация ChromaDB
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                logger.info(f"📂 Loading DB: {collection_name}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory, 
                    collection_name=collection_name, 
                    embedding_function=self.embeddings
                )
            else:
                logger.info(f"🆕 Creating DB: {collection_name}")
                self.vector_store = Chroma(
                    collection_name=collection_name, 
                    embedding_function=self.embeddings, 
                    persist_directory=self.persist_directory
                )
                # Индексируем только если это база знаний (а не база памяти юзеров)
                if collection_name == "phoneguy_brain":
                    self.index_documents()
        except Exception as e:
            logger.error(f"❌ Error initializing ChromaDB: {e}")
            self.vector_store = None

    def index_documents(self):
        """Читает файлы из папки и добавляет в базу"""
        if self.vector_store is None: return
        logger.info(f"🔍 Indexing docs from: {self.doc_path}")

        documents = []
        loaders = {
            '.pdf': PyPDFLoader, '.txt': TextLoader, '.md': UnstructuredMarkdownLoader
        }
        
        for filename in os.listdir(self.doc_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in loaders:
                filepath = os.path.join(self.doc_path, filename)
                try:
                    loader = loaders[ext](filepath)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
        
        if not documents:
            logger.warning(f"No documents found in {self.doc_path}")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vector_store.add_documents(batch)
                
        logger.info(f"✅ Indexed {len(chunks)} chunks.")

    def add_memory(self, text: str, metadata: dict):
        """Сохраняет воспоминание о звонке"""
        if not self.vector_store:
            logger.error("❌ add_memory: vector_store is None")
            return
        try:
            doc = Document(page_content=text, metadata=metadata)
            self.vector_store.add_documents([doc])
            logger.info(f"💾 Memory saved for {metadata.get('caller_id')}: {text[:50]}...")
        except Exception as e:
            logger.error(f"❌ Failed to save memory: {e}")

    def retrieve_documents(self, query: str, k: int = 2, filter_meta: dict = None) -> str:
        """Поиск с возможностью фильтрации по метаданным"""
        if not self.vector_store or not query: return ""
        
        try:
            docs = self.vector_store.similarity_search(query, k=k, filter=filter_meta)
            if not docs: return ""
            
            context_parts = []
            for d in docs:
                if 'caller_id' in d.metadata:
                    date = d.metadata.get('timestamp', '')[:10]
                    context_parts.append(f"[MEMORY {date}]: {d.page_content}")
                else:
                    src = os.path.basename(str(d.metadata.get('source', 'Unknown')))
                    context_parts.append(f"[FILE {src}]: {d.page_content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return ""

    def get_user_memories(self, caller_id: str, k: int = 5) -> str:
        """Получает все воспоминания для конкретного caller_id"""
        if not self.vector_store or not caller_id: return ""
        
        try:
            all_docs = self.vector_store.get(limit=100)
            if not all_docs or not all_docs.get('documents'):
                return ""
            
            docs = all_docs.get('documents', [])
            metadatas = all_docs.get('metadatas', [])
            
            context_parts = []
            for i, doc in enumerate(docs):
                meta = metadatas[i] if i < len(metadatas) else {}
                if meta.get('caller_id') == caller_id and meta.get('type') == 'summary':
                    date = meta.get('timestamp', '')[:10]
                    context_parts.append(f"[MEMORY {date}]: {doc}")
            
            if not context_parts:
                return ""
            
            return "\n\n".join(context_parts[:k])
        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            return ""