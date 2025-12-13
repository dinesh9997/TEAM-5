# llm1/local_llm.py

from llm1.llm_config import LLM_MODEL_NAME, TEMPERATURE, MAX_TOKENS


def get_llm():
    """
    Returns a free local LLM instance using Ollama.
    Falls back to stub if Ollama is not available.
    """
    try:
        # Try new langchain-ollama package first
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=LLM_MODEL_NAME,
                temperature=TEMPERATURE,
                num_predict=MAX_TOKENS
            )
        except ImportError:
            from langchain_community.llms import Ollama
            llm = Ollama(
                model=LLM_MODEL_NAME,
                temperature=TEMPERATURE,
                num_predict=MAX_TOKENS
            )
        
        # Quick test to see if Ollama is running
        llm.invoke("hi")
        return llm
        
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")
        print("   Using stub LLM for testing...")
        
        # Import stub from main llm module
        from llm.local_llm import _StubLLM
        return _StubLLM()
