import json
import re


def safe_parse(s: str):
    """Safely parse a JSON-like string and return a dict or the parsed value.

    If parsing fails, returns a dict with `raw` and `error` keys.
    
    Handles common LLM output issues:
    - JSON wrapped in markdown code blocks
    - Extra text before/after JSON
    - Missing or extra commas
    - Single quotes instead of double quotes
    """
    if s is None:
        return {"error": "no response", "status": "failed"}

    # Already parsed
    if isinstance(s, dict):
        return s
    
    # Convert to string if not already
    if not isinstance(s, str):
        s = str(s)
    
    # Strip whitespace
    s = s.strip()

    # Try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Remove markdown code blocks if present
    # Handles ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_match = re.search(code_block_pattern, s)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except Exception:
            pass
    
    # Extract JSON substring - find outermost { }
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = s[start:end+1]
        try:
            return json.loads(json_str)
        except Exception:
            # Try fixing common issues
            # Replace single quotes with double quotes (careful with apostrophes)
            try:
                fixed = re.sub(r"(?<![\\])'", '"', json_str)
                return json.loads(fixed)
            except Exception:
                pass
            
            # Try removing trailing commas
            try:
                fixed = re.sub(r',\s*}', '}', json_str)
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except Exception:
                pass
    
    # Return error with raw content for debugging
    return {
        "raw": s[:500] if len(s) > 500 else s,  # Limit raw content size
        "error": "Failed to parse JSON from LLM response",
        "status": "parse_failed"
    }


__all__ = ["safe_parse"]
