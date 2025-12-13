import json


def safe_parse(s: str):
    """Safely parse a JSON-like string and return a dict or the parsed value.

    If parsing fails, returns a dict with `raw` and `error` keys.
    """
    if s is None:
        return {"error": "no response"}

    # Already parsed
    if isinstance(s, dict):
        return s

    try:
        return json.loads(s)
    except Exception as e:
        # Extract JSON substring if present
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass

        return {"raw": s, "error": str(e)}


__all__ = ["safe_parse"]
