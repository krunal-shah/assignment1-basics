from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import functools
import regex as re
import time
from contextlib import contextmanager


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
num_processes = 10


@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Cooldown/Execution time: {end - start:.6f}s")


def dict_add(dictionary, key, value):
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value


def dict_subtract(dictionary, key, value):
    assert key in dictionary
    if dictionary[key] == value:
        del dictionary[key]
    else:
        dictionary[key] -= value


# The following is a serial implementation, but you can parallelize this
# by sending each start/end pair to a set of processes.
def pretokenize_chunk(index_tuple, input_file, special_tokens_bytes):
    # do the pretokenization
    start, end = index_tuple
    with open(input_file, "rb") as fr:
        fr.seek(start)
        chunk = fr.read(end - start)

    def find_end(start):
        next_end = len(chunk) + 1
        token_bytes = 0
        for special_token_bytes in special_tokens_bytes:
            token_end = chunk.find(special_token_bytes, start)
            if token_end < next_end and token_end != -1:
                next_end = token_end
                token_bytes = len(special_token_bytes)
            elif token_end == next_end and token_end != -1 and len(special_token_bytes) > token_bytes:
                next_end = token_end
                token_bytes = len(special_token_bytes)
        if next_end == len(chunk) + 1:
            return -1, 0
        return next_end, token_bytes

    token_dict = {}
    start = 0
    end, token_bytes = find_end(start)

    def process_subchunk(subchunk):
        for token in re.finditer(PAT, subchunk):
            token = token.group(0)
            dict_add(token_dict, token, 1)

    while end != -1:
        subchunk = chunk[start:end].decode("utf-8", errors="ignore")
        process_subchunk(subchunk)
        start = end + token_bytes
        end, token_bytes = find_end(start)

    process_subchunk(chunk[start:].decode("utf-8", errors="ignore"))

    return token_dict


def get_pair_counts(data):
    data_id, (key, count) = data
    counts_dict = {}
    for i in range(len(key) - 1):
        dict_add(counts_dict, key[i : i + 2], count)

    return (data_id, counts_dict)


def merge_data(inputs, max_pair, new_index):
    data_id, (data, counts) = inputs
    # this shouldn't get triggered since we only run this function on ids which have the pair, but just keeping it here
    # for sanity
    if (max_pair[0] not in data) or (max_pair[1] not in data):
        return (data_id, (data, counts), (set(), {}, {}))
    new_data = []
    pairs_to_subtract = {}
    pairs_to_delete = set()
    pairs_to_add = {}
    i = 0

    while i < len(data):
        if data[i] == max_pair[0]:
            if i < len(data) - 1 and data[i + 1] == max_pair[1]:
                new_data.append(new_index)
                i += 2
            else:
                new_data.append(data[i])
                i += 1
        else:
            new_data.append(data[i])
            i += 1
    new_data = tuple(new_data)

    _, new_pair_counts = get_pair_counts((data_id, (new_data, counts)))
    _, old_pair_counts = get_pair_counts((data_id, (data, counts)))

    for key, count in old_pair_counts.items():
        if key in new_pair_counts and new_pair_counts[key] < count:
            pairs_to_subtract[key] = count - new_pair_counts[key]
        elif key not in new_pair_counts:
            pairs_to_subtract[key] = count
            pairs_to_delete.add(key)

    for key, count in new_pair_counts.items():
        if key in old_pair_counts and old_pair_counts[key] < count:
            pairs_to_add[key] = count - old_pair_counts[key]
        elif key not in old_pair_counts:
            pairs_to_add[key] = count
    return (data_id, (new_data, counts), (pairs_to_delete, pairs_to_subtract, pairs_to_add))


def initialize_vocab(vocab, ivocab, special_tokens):
    index = 0
    for i in range(256):
        ivocab[bytes([i])] = index
        vocab[index] = bytes([i])
        index += 1
    for special_token in special_tokens:
        ivocab[special_token.encode("utf-8")] = index
        vocab[index] = special_token.encode("utf-8")
        index += 1
    return index


def pretokenized_to_vocab_ids(pretokenized_data, ivocab):
    pretokenized_data_ids = {}
    count = 0
    for key, value in pretokenized_data:
        indices = []
        key_bytes = key.encode("utf-8")
        for i in range(len(key_bytes)):
            indices.append(ivocab[key_bytes[i : i + 1]])
        pretokenized_data_ids[count] = (tuple(indices), value)
        count += 1
    return pretokenized_data_ids


def compute_max_pair(pair_counts_dict, vocab):
    max_pair = (None, 0)

    for pair, pair_count in pair_counts_dict.items():
        if pair_count > max_pair[1]:
            max_pair = (pair, pair_count)
        elif pair_count == max_pair[1] and (vocab[pair[0]], vocab[pair[1]]) > (
            vocab[max_pair[0][0]],
            vocab[max_pair[0][1]],
        ):
            max_pair = (pair, pair_count)
    return max_pair[0]


def bpe_on_pretokenized_data(pretokenized_data, vocab_size, special_tokens):
    vocab = {}  # int to bytes
    ivocab = {}  # bytes to int
    merges = []
    index = initialize_vocab(vocab, ivocab, special_tokens)

    # tokenize pretokenized dictionary using the vocab
    pretokenized_data = pretokenized_to_vocab_ids(pretokenized_data, ivocab)

    with timer("First parallel pass through the data to tokenize the data"):
        with Pool(processes=num_processes) as p:
            if num_processes == 1:
                pair_counts = [get_pair_counts((data_id, value)) for data_id, value in pretokenized_data.items()]
            else:
                pair_counts = p.imap(get_pair_counts, pretokenized_data.items(), chunksize=100)

            pair_counts_dict = {}
            pair_data_id_index = {}
            for data_id, pair_count in pair_counts:
                for pair, count in pair_count.items():
                    dict_add(pair_counts_dict, pair, count)
                    if pair in pair_data_id_index:
                        pair_data_id_index[pair].add(data_id)
                    else:
                        pair_data_id_index[pair] = {data_id}

    with timer("While loop"):
        while len(vocab) < vocab_size and pair_counts_dict:
            max_pair = compute_max_pair(pair_counts_dict, vocab)

            merge_data_func = functools.partial(merge_data, max_pair=max_pair, new_index=index)
            inputs = [(data_id, pretokenized_data[data_id]) for data_id in pair_data_id_index[max_pair]]
            merged_data = [merge_data_func(input_value) for input_value in inputs]
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            vocab[index] = vocab[max_pair[0]] + vocab[max_pair[1]]
            ivocab[vocab[index]] = index
            index += 1

            for data_id, merged, (pairs_to_delete, pairs_to_subtract, pairs_to_add) in merged_data:
                pretokenized_data[data_id] = merged
                for pair, count in pairs_to_subtract.items():
                    dict_subtract(pair_counts_dict, pair, count)
                for pair, count in pairs_to_add.items():
                    dict_add(pair_counts_dict, pair, count)
                    if pair in pair_data_id_index:
                        pair_data_id_index[pair].add(data_id)
                    else:
                        pair_data_id_index[pair] = {data_id}
                for pair in pairs_to_delete:
                    pair_data_id_index[pair].remove(data_id)
                    if not pair_data_id_index[pair]:
                        del pair_data_id_index[pair]

    return (vocab, merges)


def train_bpe(input_path, vocab_size, special_tokens):
    assert special_tokens

    pretokenized_dict = {}
    special_tokens_bytes = []
    for special_token in special_tokens:
        special_tokens_bytes.append(special_token.encode("utf-8"))

    with timer("Initial processing of file reading and creating pretokenized data"):
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
            with Pool(processes=num_processes) as p:
                pretokenized_dicts = p.imap(
                    functools.partial(
                        pretokenize_chunk,
                        input_file=f.name,
                        special_tokens_bytes=special_tokens_bytes,
                    ),
                    zip(boundaries[:-1], boundaries[1:]),
                )

                for token_dict in pretokenized_dicts:
                    for token in token_dict:
                        dict_add(pretokenized_dict, token, token_dict[token])

        pretokenized_data = pretokenized_dict.items()

    return bpe_on_pretokenized_data(pretokenized_data, vocab_size, special_tokens)


if __name__ == "__main__":
    # /Users/krunal/Documents/GitHub/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000000, ["<|endoftext|>"])
    # pretokenized_dict = {"and ": 1, " and": 5, "be ": 4, "+": 2, "before": 2}
    # bpe_on_pretokenized_data(pretokenized_dict.items(), 10000, [])
