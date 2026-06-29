import json
import re


def _balance_braces(text: str) -> str:
    """
    Attempt to auto-close missing JSON braces/brackets.
    Helps when LLM output is truncated.
    """
    open_curly = text.count("{")
    close_curly = text.count("}")
    open_square = text.count("[")
    close_square = text.count("]")

    text += "}" * max(0, open_curly - close_curly)
    text += "]" * max(0, open_square - close_square)

    return text


def safe_parse(s: str):
    """
    Robust JSON parser for LLM output.

    Handles:
    - Markdown code blocks
    - Extra text before/after JSON
    - Single quotes
    - Trailing commas
    - PARTIAL / TRUNCATED JSON (LLM output issue)
    """

    if s is None:
        return {"error": "no response", "status": "failed"}

    if isinstance(s, dict):
        return s

    if not isinstance(s, str):
        s = str(s)

    s = s.strip()

    # 1️⃣ Try direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2️⃣ Strip markdown code blocks
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except Exception:
            s = code_block.group(1).strip()

    # 3️⃣ Extract probable JSON region
    start = s.find("{")
    if start == -1:
        return {
            "raw": s[:500],
            "error": "No JSON object found",
            "status": "parse_failed"
        }

    candidate = s[start:]

    # 4️⃣ Auto-fix common issues
    candidate = _balance_braces(candidate)

    # Replace single quotes carefully
    candidate = re.sub(r"(?<!\\)'", '"', candidate)

    # Remove trailing commas
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    # 5️⃣ Final parse attempt
    try:
        return json.loads(candidate)
    except Exception as e:
        return {
            "raw": candidate[:500],
            "error": f"JSON parse failed after recovery: {str(e)}",
            "status": "parse_failed"
        }


__all__ = ["safe_parse"]
