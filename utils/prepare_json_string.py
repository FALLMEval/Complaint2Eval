import json
import re


def prepare_json_string(s):
    match = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = s.strip()

    cleaned = s.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned = cleaned.replace("\\'", "'").replace("\\`", "`").replace("\\0", "\\u0000")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parsing failed after cleaning: {e}\nInput: {cleaned}..."
        )
