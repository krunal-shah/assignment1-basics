import logging
import codecs
import functools
import pickle
from cs336_basics.tokenizer_utils import pretokenize_chunk, train_bpe, find_chunk_boundaries, timer

# from cs336_basics.pretokenization_example import find_chunk_boundaries
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from itertools import batched
import argparse
import torch

NUM_PROCESSES = 10

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--vocab_size", type=int)

    return parser.parse_args()


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = set(special_tokens or [])
        self.special_tokens_bytes = set()
        self.ivocab = {}
        index = 0

        for tid, token_bytes in vocab.items():
            self.ivocab[token_bytes] = tid
            if tid > index:
                index = tid
        index += 1
        self.index = index

        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            self.special_tokens_bytes.add(special_token_bytes)
            if special_token_bytes not in self.ivocab:
                self.ivocab[special_token_bytes] = self.index
                self.vocab[self.index] = special_token_bytes
                self.index += 1

        self.merge_index = {}

        for i, (token1, token2) in enumerate(self.merges):
            if token1 in self.merge_index:
                self.merge_index[token1][token2] = i
            else:
                self.merge_index[token1] = {token2: i}

    def save(self, output_file):
        saved_state = {"vocab": self.vocab, "merges": self.merges, "special_tokens": self.special_tokens}
        with open(output_file, "wb") as f:
            pickle.dump(saved_state, f)

    @staticmethod
    def load(tokenizer_file, special_tokens=[]):
        with open(Path(tokenizer_file), "rb") as f:
            tokenizer_data = pickle.load(f)
        tokenizer_data["special_tokens"] = list(set(special_tokens + list(tokenizer_data["special_tokens"])))
        return Tokenizer(**tokenizer_data)

    def encode_chunk(self, chunk: str | bytes):
        chunk_bytes = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        if chunk in self.special_tokens:
            assert chunk_bytes in self.ivocab
            return [chunk_bytes]
        else:
            chunk_bytes_list = []
            for i in range(len(chunk_bytes)):
                chunk_bytes_list.append(chunk_bytes[i : i + 1])

        merged = True
        while merged:
            min_merge = len(self.merges) + 1
            for tk1, tk2 in zip(chunk_bytes_list[:-1], chunk_bytes_list[1:]):
                if tk1 in self.merge_index and tk2 in self.merge_index[tk1]:
                    min_merge = min(min_merge, self.merge_index[tk1][tk2])

            if min_merge != len(self.merges) + 1:
                merged_chunked_bytes_list = []
                index = 0
                while index < len(chunk_bytes_list) - 1:
                    if (chunk_bytes_list[index], chunk_bytes_list[index + 1]) == self.merges[min_merge]:
                        merged_chunked_bytes_list.append(chunk_bytes_list[index] + chunk_bytes_list[index + 1])
                        index += 2
                    else:
                        merged_chunked_bytes_list.append(chunk_bytes_list[index])
                        index += 1
                if index == len(chunk_bytes_list) - 1:
                    merged_chunked_bytes_list.append(chunk_bytes_list[-1])
                chunk_bytes_list = merged_chunked_bytes_list
            else:
                merged = False

        return chunk_bytes_list

    def encode(self, text: str | bytes):
        special_tokens = self.special_tokens_bytes if isinstance(text, bytes) else self.special_tokens
        pretokenized_chunks = pretokenize_chunk(text, special_tokens, return_dict=False)
        tokenized_text = [self.ivocab[token] for chunk in pretokenized_chunks for token in self.encode_chunk(chunk)]
        return tokenized_text

    def encode_iterable(self, iterable):
        combined_string = ""
        for string in iterable:
            special_token_present = False
            for special_token in self.special_tokens:
                if special_token in string:
                    special_token_present = True
                    special_token_position = string.find(special_token)
                    combined_string += string[:special_token_position]
                    string = string[special_token_position:]
                    break

            if special_token_present:
                yield from self.encode(combined_string)
                combined_string = string
            else:
                combined_string += string

        if combined_string:
            yield from self.encode(combined_string)

    def encode_iterable_bytes(self, iterable):
        combined_string = b""
        for string in iterable:
            special_token_present = False
            borrowed_string_len = min(len(combined_string), 40)
            borrowed_string = combined_string[-1 * borrowed_string_len :]
            string = borrowed_string + string
            combined_string = combined_string[: -1 * borrowed_string_len]
            for special_token_bytes in self.special_tokens_bytes:
                if string.find(special_token_bytes) != -1:
                    special_token_present = True
                    special_token_position = string.find(special_token_bytes)
                    combined_string += string[:special_token_position]
                    string = string[special_token_position:]
                    break

            if special_token_present:
                yield from self.encode(combined_string)
                combined_string = string
            else:
                combined_string += string
        if combined_string:
            yield from self.encode(combined_string)

    def iterate_file_chunked(self, index_tuple, filepath):
        start, end = index_tuple
        with open(filepath, "rb") as fr:
            fr.seek(start)
            bytes_to_read = end - start
            while bytes_to_read:
                chunk = fr.read(min(4096, bytes_to_read))
                bytes_to_read -= len(chunk)
                yield chunk

    def encode_file_chunk(self, index_tuple, filepath):
        encoded_ids = [token for token in self.encode_iterable_bytes(self.iterate_file_chunked(index_tuple, filepath))]
        return encoded_ids

    def encode_file(self, filepath, outfilepath):
        with timer("Encode file"):
            with open(filepath, "rb") as f:
                boundaries = find_chunk_boundaries(f, NUM_PROCESSES * 10, self.special_tokens_bytes)
                encode_file_chunk = functools.partial(self.encode_file_chunk, filepath=filepath)
                with Pool(processes=NUM_PROCESSES) as workers:
                    token_iterators_list = workers.imap(encode_file_chunk, zip(boundaries[:-1], boundaries[1:]))

                    with open(outfilepath, "wb") as outfile:
                        for token_iterator in token_iterators_list:
                            for chunk in batched(token_iterator, 4096):
                                np.asarray(chunk, dtype=np.int32).tofile(outfile)

    def bytes_to_utf_iterable(self, chunk_iterable):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for chunk in chunk_iterable:
            yield decoder.decode(chunk)
        yield decoder.decode(b"", final=True)

    def iterate_np_array_file_chunked(self, filepath):
        with open(filepath, "rb") as f:
            while True:
                chunk = np.fromfile(f, dtype=np.int32, count=4096)
                if chunk.size == 0:
                    break
                yield self.decode(chunk.tolist(), to_utf=False)

    def decode_file(self, filepath, outfilepath):
        with timer("Decode file"):
            with open(outfilepath, "w", encoding="utf-8") as outfile:
                for utf in self.bytes_to_utf_iterable(self.iterate_np_array_file_chunked(filepath)):
                    outfile.write(utf)

    def decode(self, ids, to_utf=True):
        decoded_bytes = [self.vocab[tid] for tid in ids]
        decoded_bytes = b"".join(decoded_bytes)

        return decoded_bytes.decode("utf-8", errors="replace") if to_utf else decoded_bytes


def get_training_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    max_valid_index = n - context_length - 1
    sampled_indices = np.random.randint(0, high=max_valid_index + 1, size=batch_size)
    batch_indices = np.expand_dims(np.arange(context_length + 1), axis=0)
    batch_indices = np.expand_dims(sampled_indices, axis=-1) + batch_indices
    batch_indices_flattened = batch_indices.reshape(-1)
    batch = np.take(dataset, axis=0, indices=batch_indices_flattened).reshape(batch_size, context_length + 1)

    batch = torch.tensor(batch, device=device)
    return (batch[:, :-1], batch[:, 1:])


def get_validation_batch_itr(dataset, batch_size, context_length, device):
    n = len(dataset)
    max_valid_index = n - context_length - 1

    for i in range(0, max_valid_index + 1, context_length * batch_size):
        sampled_indices = np.arange(i, min(max_valid_index + 1, i + batch_size * context_length), step=context_length)
        current_batch_size = len(sampled_indices)
        batch_indices = np.expand_dims(np.arange(context_length + 1), axis=0)
        batch_indices = np.expand_dims(sampled_indices, axis=-1) + batch_indices
        batch_indices_flattened = batch_indices.reshape(-1)
        batch = np.take(dataset, axis=0, indices=batch_indices_flattened).reshape(
            current_batch_size, context_length + 1
        )

        batch = torch.tensor(batch, device=device)
        yield (batch[:, :-1], batch[:, 1:])


if __name__ == "__main__":
    args = parse_args()
    if args.datapath and not args.tokenizer_path:
        filepath = Path(args.datapath)
        vocab_size = args.vocab_size
        special_tokens = ["<|endoftext|>"]
        vocab, merges = train_bpe(filepath, vocab_size, special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        tokenizer.save(f"data/{filepath.stem}_tokenizer.pkl")
    if args.tokenizer_path and args.datapath:
        tokenizer = Tokenizer.load(args.tokenizer_path)
        filepath = Path(args.datapath)
        encoded_filepath = Path(filepath.with_suffix(".encoded"))
        decoded_filepath = Path(filepath.with_suffix(".decoded"))
        # with open(filepath, "rb") as f:
        #     tokenized_data = []
        #     iterator_over_bytes = iter(lambda: f.read(10000000), b"")
        #     with timer("Encoding time"):
        #         tokenized_data += tokenizer.encode_iterable_bytes(iterator_over_bytes)
        # print(f"{len(tokenized_data)=}")
        # decoded_data = tokenizer.decode(tokenized_data)

        # with open(filepath, "rb") as f:
        #     actual_data = f.read().decode("utf-8", errors="replace")

        # print(f"{len(decoded_data)}, {len(actual_data)}, {actual_data == decoded_data}")

        tokenizer.encode_file(filepath, encoded_filepath)
        tokenizer.decode_file(encoded_filepath, decoded_filepath)

        # print(f"{actual_data=}")
        # print(f"{decoded_data=}")

    # filepath = Path(sys.argv[1])
    # special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe(filepath, 10000, special_tokens)
    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    # tokenizer.save(f"data/{filepath.stem}_tokenizer.pkl")

    # vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    # merges = [
    #     (b"t", b"h"),
    #     (b" ", b"c"),
    #     (b" ", b"a"),
    #     (b"th", b"e"),
    #     (b" a", b"t"),
    # ]

    # special_tokens = ["<abc>"]

    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    # print(tokenizer.encode("the<abc> cat<abc> ate"))
