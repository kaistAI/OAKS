import json
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path}")
    return data


def load_records(path: str) -> Iterable[Dict[str, Any]]:
    """Load a corpus file that can be JSON array, JSON object, or JSONL.

    Normalizes records into dictionaries. If the top-level object is a dict mapping
    book_id to a list of chunks, yields records like:
      {"book_id": <id>, "chunk_index": <i>, "text": <chunk_text>, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON is '[' but not a list")
            for item in data:
                if isinstance(item, dict):
                    yield item
                else:
                    yield {"text": str(item)}
            return
        if first_char == "{":
            obj = json.load(f)
            if not isinstance(obj, dict):
                raise ValueError("Top-level JSON is '{' but not a dict")
            # {book_id: [chunk0, chunk1, ...]}
            is_dict_of_lists = all(isinstance(v, list) for v in obj.values()) if obj else False
            if is_dict_of_lists:
                for book_id, chunks in obj.items():
                    for idx, chunk in enumerate(chunks):
                        if isinstance(chunk, dict):
                            rec = {"book_id": str(book_id), "chunk_index": idx, **chunk}
                            yield rec
                        else:
                            yield {"book_id": str(book_id), "chunk_index": idx, "text": str(chunk)}
                return
            # Otherwise, yield values as records, attaching their keys when helpful
            for key, value in obj.items():
                if isinstance(value, dict):
                    rec = {"id": key, **value}
                    yield rec
                else:
                    yield {"id": key, "text": str(value)}
            return
        # Fallback: treat as JSONL
        for obj in _iter_jsonl(path):
            yield obj


def pick_text_field(record: Dict[str, Any]) -> Tuple[str, str]:
    """Return (field_name, text_value) for the main text field in a corpus record.

    Tries common keys; raises if none found.
    """
    candidates = [
        "text",
        "chunk_text",
        "content",
        "passage",
        "body",
        "paragraph",
    ]
    for key in candidates:
        if key in record and isinstance(record[key], str):
            return key, record[key]
    # Fallback: try to find the first string value
    for key, value in record.items():
        if isinstance(value, str) and len(value) > 0:
            return key, value
    raise KeyError("No text-like field found in record")


def pick_question_field(record: Dict[str, Any]) -> Tuple[str, str]:
    candidates = ["question", "query", "prompt"]
    for key in candidates:
        if key in record and isinstance(record[key], str):
            return key, record[key]
    # Fallback
    for key, value in record.items():
        if isinstance(value, str) and value.endswith("?"):
            return key, value
    raise KeyError("No question-like field found in record") 