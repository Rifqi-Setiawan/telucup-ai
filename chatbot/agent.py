import logging
import time

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .config import GEMINI_MODEL, require_google_api_key
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

_agent = None


def get_agent():
    """Singleton agent builder."""
    global _agent

    if _agent is None:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=require_google_api_key(),
            temperature=0.2,
            max_output_tokens=1024,
            timeout=25,
        )
        _agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )

    return _agent


def run_agent(question: str) -> str:
    """
    Jalankan agent secara stateless untuk satu pertanyaan.
    Tidak ada history yang disimpan antar pemanggilan.
    """
    started_at = time.perf_counter()
    logger.info("[CHATBOT] Incoming question length=%s", len(question))

    agent = get_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
    })
    final_message = result["messages"][-1]

    logger.info("[CHATBOT] Completed in %.2fs", time.perf_counter() - started_at)
    return final_message.content
