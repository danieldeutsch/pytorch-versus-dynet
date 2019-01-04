import argparse
import json
from typing import Any, Dict, List

bos = '<bos>'
eos = '<eos>'
unk = '<unk>'


def load_src(filename: str) -> List[List[str]]:
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            index = line.find('(')
            root = line[:index - 1].replace(' ', '')
            transformation = line[index:].replace(' ', '_')
            source = [bos] + list(root) + [eos] + [transformation]
            data.append(source)
    return data


def load_tgt(filename: str) -> List[List[str]]:
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().replace(' ', '')
            target = [bos] + list(line) + [eos]
            data.append(target)
    return data


def make_vocab(src: List[List[str]], tgt: List[List[str]]) -> Dict[str, int]:
    token2id = {}
    token2id[bos] = 0
    token2id[eos] = 1
    token2id[unk] = 2
    for word in src + tgt:
        for token in word:
            if token not in token2id:
                token2id[token] = len(token2id)
    return token2id


def encode(src: List[str], tgt: List[str], token2id: Dict[str, int]) -> List[Dict[str, Any]]:
    encoded = []
    for source, target in zip(src, tgt):
        source_encoded = []
        for token in source:
            if token in token2id:
                source_encoded.append(token2id[token])
            else:
                source_encoded.append(token2id[unk])

        target_encoded = []
        for token in target:
            if token in token2id:
                target_encoded.append(token2id[token])
            else:
                target_encoded.append(token2id[unk])

        data = {
            'source_string': ''.join(source),
            'target_string': ''.join(target),
            'source': source_encoded,
            'target': target_encoded
        }
        encoded.append(data)
    return encoded


def save(data: List[Dict[str, Any]], filename: str) -> None:
    with open(filename, 'w') as out:
        for item in data:
            out.write(json.dumps(item) + '\n')


def save_vocab(token2id: Dict[str, int], filename: str) -> None:
    with open(filename, 'w') as out:
        out.write(json.dumps(token2id))


def main(args):
    train_src = load_src(args.train_src)
    train_tgt = load_tgt(args.train_tgt)
    valid_src = load_src(args.valid_src)
    valid_tgt = load_tgt(args.valid_tgt)
    test_src = load_src(args.test_src)
    test_tgt = load_tgt(args.test_tgt)

    token2id = make_vocab(train_src, train_tgt)

    train = encode(train_src, train_tgt, token2id)
    valid = encode(valid_src, valid_tgt, token2id)
    test = encode(test_src, test_tgt, token2id)

    save(train, args.train_output)
    save(valid, args.valid_output)
    save(test, args.test_output)

    save_vocab(token2id, args.vocab_output)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('train_src')
    argp.add_argument('train_tgt')
    argp.add_argument('train_output')
    argp.add_argument('valid_src')
    argp.add_argument('valid_tgt')
    argp.add_argument('valid_output')
    argp.add_argument('test_src')
    argp.add_argument('test_tgt')
    argp.add_argument('test_output')
    argp.add_argument('vocab_output')
    args = argp.parse_args()
    main(args)
