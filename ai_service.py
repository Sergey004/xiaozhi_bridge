import logging
import rich.logging
from dotenv import load_dotenv
import os
import re
import random
import asyncio
import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ai_core.document_processor import DocumentProcessor
from ai_core.ai_config import DEFAULT_PROMPT

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:
    ChatNVIDIA = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

SYSTEM_INSTRUCTIONS = "/no_think\n\n" + DEFAULT_PROMPT

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- КОНФИГУРАЦИЯ ---
ALLOWED_TTS_TAGS = {
    "[clear throat]",
    "[sigh]",
    "[shush]",
    "[cough]",
    "[groan]",
    "[sniff]",
    "[gasp]",
    "[chuckle]",
    "[laugh]",
}
FALLBACK_PHRASES = [
    "Uh, hello? I think the signal is breaking up.",
    "Uh, sorry, could you say that again?",
    "Um, I didn't catch that.",
]

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

AI_ENABLED = False
llm = None
rag_processor = None
memory_processor = None
current_provider = None


def _init_llm():
    global llm, AI_ENABLED, current_provider

    if LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            logger.error(
                "❌ langchain-ollama not installed. Run: pip install langchain-ollama"
            )
            return False

        try:
            llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                num_predict=LLM_MAX_TOKENS,
            )
            current_provider = "ollama"
            logger.info(f"✅ Ollama initialized: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
            logger.info("🔥 Warming up LLM...")
            llm.invoke([HumanMessage(content="/no_think hi")])
            logger.info("✅ LLM warm.")
        except Exception as e:
            logger.error(f"❌ Error initializing Ollama: {e}")
            return False

    else:
        if not NVIDIA_API_KEY or ChatNVIDIA is None:
            logger.error(
                "❌ NVIDIA API key not set or langchain-nvidia-ai-endpoints not installed"
            )
            return False

        try:
            llm = ChatNVIDIA(
                api_key=NVIDIA_API_KEY,
                model=os.getenv("NVIDIA_MODEL", "meta/llama3-70b-instruct"),
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                model_kwargs={
                    "extra_body": {"chat_template_kwargs": {"thinking": False}}
                },
            )
            current_provider = "nvidia"
            logger.info("✅ NVIDIA AI initialized")
            logger.info("🔥 Warming up LLM...")
            llm.invoke([HumanMessage(content="/no_think hi")])
            logger.info("✅ LLM warm.")
        except Exception as e:
            logger.error(f"❌ Error initializing NVIDIA AI: {e}")
            return False
        
    return True


def _init_memory():
    global rag_processor, memory_processor

    try:
        rag_processor = DocumentProcessor(
            doc_path="knowledge_base", collection_name="phoneguy_brain"
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize rag_processor: {e}")
        rag_processor = None

    try:
        if not os.path.exists("user_memories"):
            os.makedirs("user_memories")
        memory_processor = DocumentProcessor(
            doc_path="user_memories", collection_name="phoneguy_users"
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize memory_processor: {e}")
        memory_processor = None


if _init_llm():
    _init_memory()
    AI_ENABLED = True
    logger.info("✅ AI & Memory Systems Online")

conversation_history = []
current_user_context = ""

_memory_banks = {}


def set_memory_bank(character_name: str):
    """Устанавливает отдельную базу памяти для персонажа"""
    global memory_processor
    bank_path = f"memories_{character_name}"

    if character_name in _memory_banks:
        memory_processor = _memory_banks[character_name]
        logger.info(f"📂 Switched to memory bank: {bank_path}")
        return

    if not os.path.exists(bank_path):
        os.makedirs(bank_path)

    bank = DocumentProcessor(
        doc_path=bank_path, collection_name=f"{character_name}_users"
    )
    _memory_banks[character_name] = bank
    memory_processor = bank
    logger.info(f"📂 Created/Switched to memory bank: {bank_path}")


def set_caller_context(caller_id: str):
    """Загружает досье на звонящего"""
    global current_user_context
    current_user_context = ""

    if memory_processor and caller_id:
        logger.info(f"🧠 Retrieving memory for CallerID: {caller_id}")
        memories = memory_processor.get_user_memories(caller_id, k=5)

        if memories:
            current_user_context = f"\n\n[MEMORY - PREVIOUS CALLS WITH THIS GUARD]:\n{memories}\n(If the user seems familiar, acknowledge it nervously. If not, ignore this.)"
            logger.info("🧠 Memory loaded.")
        else:
            logger.info("🧠 No memory found (New Caller).")


def _build_adaptive_system_prompt(
    context_type: str = "normal", custom_prompt: Optional[str] = None
) -> str:
    """
    Создаёт адаптивный system prompt в зависимости от контекста.

    Args:
        context_type: 'greeting' | 'normal' | 'ongoing'
        custom_prompt: Кастомный system prompt для персонажа
    """
    base = custom_prompt if custom_prompt else SYSTEM_INSTRUCTIONS

    if current_user_context:
        base += f"\n\n=== KNOWN CALLER ==={current_user_context}\nYou RECOGNIZE this person. Greet them by name warmly but nervously."
    else:
        base += "\n\n=== NEW CALLER ===\nYou do NOT recognize this caller. Be cautious and suspicious. Ask 'Uh, hello? Who is this?'"

    if context_type == "greeting":
        base += "\n\nFIRST CONTACT: Your greeting just played. Wait for their response."
    elif context_type == "ongoing":
        base += "\n\nCONVERSATION ACTIVE: Respond naturally to what they say. Reference previous calls if you know them."
    elif context_type == "normal":
        base += "\n\nRespond to the user's message naturally."

    return base


def generate_phoneguy_greeting(
    custom_system_prompt: Optional[str] = None,
    custom_greeting_prompt: Optional[str] = None,
) -> str:
    """
    Генерирует приветствие для первого контакта.
    Правильно инициализирует conversation_history без дублирования сообщений.

    Args:
        custom_system_prompt: Кастомный system prompt персонажа
        custom_greeting_prompt: Кастомный промпт для генерации приветствия
    """
    if not AI_ENABLED or not llm:
        logger.warning(
            "⚠️ AI not enabled or llm not initialized, using fallback greeting"
        )
        return "Uh, hello? Hello, hello?"

    logger.info("🎯 Generating adaptive greeting...")

    full_system = _build_adaptive_system_prompt(
        context_type="greeting", custom_prompt=custom_system_prompt
    )
    conversation_history.append(SystemMessage(content=full_system))

    default_greeting_prompt = (
        "/no_think\n Someone just called you. Say hello in your nervous but friendly way."
    )
    greeting_prompt = (
        custom_greeting_prompt if custom_greeting_prompt else default_greeting_prompt
    )

    conversation_history.append(HumanMessage(content=greeting_prompt))

    try:
        response = llm.invoke(conversation_history)
        raw = response.content.strip() if response.content else ""
        cleaned = clean_response_text(raw)

        if cleaned:
            conversation_history.append(AIMessage(content=cleaned))
            logger.info(f"🤖 Greeting generated: {cleaned[:50]}...")
            return cleaned
    except Exception as e:
        logger.error(f"❌ Greeting generation failed: {e}")

    fallback = "Uh, hello? Hello, hello?"
    conversation_history.append(AIMessage(content=fallback))
    return fallback


async def summarize_and_save(caller_id: str):
    """Сжимает диалог и сохраняет в память"""
    logger.info(f"📝 [summarize_and_save] Called for caller_id: {caller_id}")
    logger.info(
        f"📝 [summarize_and_save] memory_processor exists: {memory_processor is not None}"
    )
    logger.info(
        f"📝 [summarize_and_save] conversation_history length: {len(conversation_history)}"
    )

    if not memory_processor:
        logger.error(
            "❌ memory_processor is None - NVIDIA_API_KEY missing or initialization failed"
        )
        return
    if not conversation_history:
        logger.warning("⚠️ conversation_history is empty - nothing to save")
        return
    if not caller_id:
        logger.warning("⚠️ caller_id is empty")
        return

    logger.info("📝 Summarizing call...")

    dialog_text = "\n".join(
        [
            f"{m.type}: {m.content}"
            for m in conversation_history
            if isinstance(m, (HumanMessage, AIMessage))
        ]
    )

    summary_prompt = f"""Analyze this call between Phone Guy and a Guard (User).
Write a short summary (in English) of 2-3 sentences.
Focus on: Name (if given), current Night, specific animatronics discussed, user's emotional state.

DIALOGUE:
{dialog_text[-4000:]}

SUMMARY:
"""

    try:
        res = await asyncio.to_thread(llm.invoke, summary_prompt)
        summary = res.content.strip()

        timestamp = datetime.datetime.now().isoformat()
        meta = {"caller_id": caller_id, "timestamp": timestamp, "type": "summary"}

        memory_processor.add_memory(summary, meta)
        logger.info(f"💾 Memory saved: {summary}")
    except Exception as e:
        logger.error(f"❌ Summarization failed: {e}")


def clean_response_text(text):
    if not text:
        return ""
    text = re.sub(r"\*+[^\*]+\*+", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    def replace_bracket(match):
        tag = match.group(0).lower()
        for allowed in ALLOWED_TTS_TAGS:
            if allowed in tag:
                return match.group(0)
        return ""

    text = re.sub(r"\[.*?\]", replace_bracket, text)
    return re.sub(r"\s+", " ", text).strip()


def phoneguy_reply(user_text: str, ignore_system_instructions: bool = False) -> str:
    if not AI_ENABLED or not llm:
        return "Uh, hello?"
    if not user_text or len(user_text.strip()) < 2:
        return None

    rag_context = ""
    if rag_processor:
        found_text = rag_processor.retrieve_documents(user_text, k=1, filter_meta=None)
        if found_text:
            rag_context = f"\n\n[OFFICE FILES]:\n{found_text}\n(Use this info, act like you're reading it)"

    if not ignore_system_instructions:
        if not conversation_history:
            full_system = _build_adaptive_system_prompt(context_type="normal")
            conversation_history.append(SystemMessage(content=full_system))
        elif len(conversation_history) == 2:
            system_msg = conversation_history[0]
            if isinstance(system_msg, SystemMessage):
                adaptive_system = _build_adaptive_system_prompt(context_type="ongoing")
                conversation_history[0] = SystemMessage(content=adaptive_system)
                logger.info("🔄 Updated system prompt for ongoing conversation")

    full_msg = user_text + rag_context
    conversation_history.append(HumanMessage(content=full_msg))

    max_retries = 2
    ai_response_text = ""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(conversation_history)
            raw = response.content.strip() if response.content else ""
            cleaned = clean_response_text(raw)
            if cleaned:
                ai_response_text = cleaned
                break
        except Exception:
            pass

    if not ai_response_text:
        ai_response_text = random.choice(FALLBACK_PHRASES)
        conversation_history.pop()
    else:
        conversation_history.append(AIMessage(content=ai_response_text))
        conversation_history[-2] = HumanMessage(content=user_text)

    if len(conversation_history) > 12:
        conversation_history[:] = conversation_history[-12:]
    return ai_response_text


def custom_reply(user_text: str, custom_prompt: Optional[str] = None) -> str:
    """Ответ для кастомного персонажа. Не ломает phoneguy_reply."""
    if not AI_ENABLED or not llm:
        return "Uh, hello?"
    if not user_text or len(user_text.strip()) < 2:
        return None

    if not conversation_history:
        full_system = _build_adaptive_system_prompt(
            context_type="normal", custom_prompt=custom_prompt
        )
        conversation_history.append(SystemMessage(content=full_system))
    elif len(conversation_history) == 2:
        system_msg = conversation_history[0]
        if isinstance(system_msg, SystemMessage):
            adaptive_system = _build_adaptive_system_prompt(
                context_type="ongoing", custom_prompt=custom_prompt
            )
            conversation_history[0] = SystemMessage(content=adaptive_system)
            logger.info("🔄 Updated system prompt for ongoing conversation")

    conversation_history.append(HumanMessage(content=user_text))

    max_retries = 2
    ai_response_text = ""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(conversation_history)
            raw = response.content.strip() if response.content else ""
            cleaned = clean_response_text(raw)
            if cleaned:
                ai_response_text = cleaned
                break
        except Exception:
            pass

    if not ai_response_text:
        ai_response_text = "Huh? What'd you say?"
        conversation_history.pop()
    else:
        conversation_history.append(AIMessage(content=ai_response_text))
        conversation_history[-2] = HumanMessage(content=user_text)

    if len(conversation_history) > 12:
        conversation_history[:] = conversation_history[-12:]
    return ai_response_text


def reset_conversation_history():
    conversation_history.clear()


def switch_llm_provider(provider: str) -> bool:
    """
    Переключает LLM провайдера между 'nvidia' и 'ollama'.
    Сбрасывает историю разговора при переключении.

    Args:
        provider: 'nvidia' или 'ollama'

    Returns:
        True если переключение успешно, False иначе
    """
    global llm, current_provider

    provider = provider.lower()
    if provider not in ("nvidia", "ollama"):
        logger.error(f"❌ Unknown provider: {provider}. Use 'nvidia' or 'ollama'")
        return False

    if provider == current_provider:
        logger.info(f"ℹ️ Already using {provider}")
        return True

    logger.info(f"🔄 Switching LLM provider to {provider}...")

    conversation_history.clear()
    current_provider = None
    llm = None

    if provider == "ollama":
        if ChatOllama is None:
            logger.error(
                "❌ langchain-ollama not installed. Run: pip install langchain-ollama"
            )
            return False
        try:
            llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                num_predict=LLM_MAX_TOKENS,
            )
            current_provider = "ollama"
            logger.info(f"✅ Switched to Ollama: {OLLAMA_MODEL}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to switch to Ollama: {e}")
            return False

    else:
        if not NVIDIA_API_KEY or ChatNVIDIA is None:
            logger.error(
                "❌ NVIDIA API key not set or langchain-nvidia-ai-endpoints not installed"
            )
            return False
        try:
            llm = ChatNVIDIA(
                api_key=NVIDIA_API_KEY,
                model=os.getenv("NVIDIA_MODEL", "meta/llama3-70b-instruct"),
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                model_kwargs={
                    "extra_body": {"chat_template_kwargs": {"thinking": True}}
                },
            )
            current_provider = "nvidia"
            logger.info("✅ Switched to NVIDIA")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to switch to NVIDIA: {e}")
            return False


def get_llm_provider() -> str:
    """Возвращает текущего провайдера."""
    return current_provider if current_provider else "unknown"


def is_ollama_available() -> bool:
    """Проверяет доступность Ollama."""
    if ChatOllama is None:
        return False
    import httpx

    try:
        response = httpx.get(OLLAMA_BASE_URL + "/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False
