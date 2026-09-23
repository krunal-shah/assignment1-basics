import pickle
import sys
from tokenizer_utils import train_bpe
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def save(self, output_file):
        saved_state = {"vocab": self.vocab, "merges": self.merges, "special_tokens": self.special_tokens}
        with open(output_file, "wb") as f:
            pickle.dump(saved_state, f)


if __name__ == "__main__":
    filepath = Path(sys.argv[1])
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(filepath, 10000, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(f"data/{filepath.stem}_tokenizer.pkl")
