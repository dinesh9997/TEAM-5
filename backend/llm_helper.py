"""Helper to load the project's LLM implementation.

Loads the LLM instance from `llm/local_llm.py` which provides
a Google Gemini-backed LLM with automatic fallback to a stub.

If the standard import fails (e.g., because a package named `llm`
shadows the local folder), it loads the file directly.
"""
import importlib
import importlib.util
import os


def _load_local_llm_from_file():
    """Load llm/local_llm.py directly from the filesystem."""
    path = os.path.join(os.path.dirname(__file__), "llm", "local_llm.py")
    if not os.path.exists(path):
        raise ModuleNotFoundError("LLM implementation not found at %s" % path)

    spec = importlib.util.spec_from_file_location("_local_llm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "llm")


try:
    # Prefer the package import if it works and contains `local_llm` module
    llm_mod = importlib.import_module("llm.local_llm")
    llm = getattr(llm_mod, "llm")
except Exception:
    # Fallback to loading the local file directly
    llm = _load_local_llm_from_file()


__all__ = ["llm"]
