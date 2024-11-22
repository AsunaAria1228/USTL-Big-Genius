import gzip
import html
import os
from functools import lru_cache
import ftfy
import regex as re


@lru_cache()
def default_bpe_path():
    """Get default BPE file path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Map utf-8 byte values to unique Unicode characters.
    This ensures reversibility and avoids whitespace/control characters.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):  # Iterate through all possible byte values
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return the set of symbol pairs in a word.
    The word is represented as a tuple of symbols.
    """
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def basic_clean(text):
    """Perform basic text cleaning."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean and normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


class SimpleTokenizer:
    def __init__(self, bpe_path: str = default_bpe_path()):
        # Byte-level encoding and decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Load and process BPE merge rules
        self.merges = self._load_merges(bpe_path)
        self.bpe_ranks = dict(zip(self.merges, range(len(self.merges))))

        # Vocabulary and encoding/decoding mappings
        vocab = list(self.byte_encoder.values()) + [v + '</w>' for v in self.byte_encoder.values()]
        vocab.extend([''.join(merge) for merge in self.merges])
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.decoder = {i: v for v, i in self.encoder.items()}

        # BPE caching
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # Tokenization pattern
        self.token_pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+",
            re.IGNORECASE
        )

    def _load_merges(self, bpe_path):
        """Load BPE merge rules from the file."""
        with gzip.open(bpe_path, 'rt', encoding='utf-8') as f:
            merges = f.read().splitlines()
        return [tuple(merge.split()) for merge in merges[1:49152 - 256 - 2 + 1]]

    def bpe(self, token):
        """Apply BPE encoding to a single token."""
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = get_pairs(word)

        encoded_word = ' '.join(word)
        self.cache[token] = encoded_word
        return encoded_word

    def encode(self, text):
        """Encode text into BPE tokens."""
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.token_pattern, text):
            encoded_token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(encoded_token).split(' '))
        return tokens

    def decode(self, tokens):
        """Decode BPE tokens back into text."""
        text = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text.replace('</w>', ' ')
        