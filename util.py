import json
from typing import Any, Dict, List


def load_data(filename: str) -> List[Dict[str, Any]]:
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_vocab(filename: str) -> Dict[str, int]:
    with open(filename, 'r') as f:
        return json.loads(f.read())


def get_vocab_size(token2id: Dict[str, int]) -> int:
    max_token_id = 0
    for token_id in token2id.values():
        max_token_id = max(max_token_id, token_id)
    vocab_size = max_token_id + 1
    return vocab_size
