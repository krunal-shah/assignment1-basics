from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import functools
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
num_processes = 1


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
def pretokenize_chunk(index_tuple, input_file, special_tokens, special_tokens_bytes):
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
        if next_end == len(chunk) + 1:
            return -1, 0
        return next_end, token_bytes

    token_dict = {}
    # token_dict["number_of_subchunks"] = 1
    start = 0
    end, token_bytes = find_end(start)

    expected_splits = 1
    for special_token in special_tokens:
        expected_splits += chunk.decode("utf-8", errors="ignore").count(special_token)

    def process_subchunk(subchunk):
        for token in re.finditer(PAT, subchunk):
            token = token.group(0)
            dict_add(token_dict, token, 1)

    while end != -1:
        subchunk = chunk[start:end].decode("utf-8", errors="ignore")
        process_subchunk(subchunk)
        start = end + token_bytes
        end, token_bytes = find_end(start)
        # token_dict["number_of_subchunks"] += 1

    # assert token_dict["number_of_subchunks"] == expected_splits, (
    #     f"{token_dict['number_of_subchunks']}, {expected_splits}"
    # )

    process_subchunk(chunk[start:].decode("utf-8", errors="ignore"))

    return token_dict


def get_pair_counts(data):
    key, count = data
    counts_dict = {}
    for i in range(len(key) - 1):
        dict_add(counts_dict, key[i : i + 2], count)

    return counts_dict


def merge_data(tokenized_data, max_pair, new_index):
    data, counts = tokenized_data
    if (max_pair[0] not in data) or (max_pair[1] not in data):
        return ((data, counts), ({}, {}))
    new_data = []
    pairs_to_remove = {}
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

    new_pair_counts = get_pair_counts((new_data, counts))
    old_pair_counts = get_pair_counts((data, counts))

    for key, count in old_pair_counts.items():
        if key in new_pair_counts and new_pair_counts[key] < count:
            pairs_to_remove[key] = count - new_pair_counts[key]
        elif key not in new_pair_counts:
            pairs_to_remove[key] = count

    for key, count in new_pair_counts.items():
        if key in old_pair_counts and old_pair_counts[key] < count:
            pairs_to_add[key] = count - old_pair_counts[key]
        elif key not in old_pair_counts:
            pairs_to_add[key] = count
    return ((new_data, counts), (pairs_to_remove, pairs_to_add))


def bpe_on_pretokenized_data(pretokenized_data, vocab_size, special_tokens):
    vocab = {}  # int to bytes
    ivocab = {}  # bytes to int
    merges = []
    index = 0
    for special_token in special_tokens:
        ivocab[special_token.encode("utf-8")] = index
        vocab[index] = special_token.encode("utf-8")
        index += 1
    for i in range(256):
        ivocab[bytes([i])] = index
        vocab[index] = bytes([i])
        index += 1

    # tokenize pretokenized dictionary using the vocab
    pretokenized_data_ids = []
    for key, value in pretokenized_data:
        indices = []
        key_bytes = key.encode("utf-8")
        for i in range(len(key_bytes)):
            indices.append(ivocab[key_bytes[i : i + 1]])
        pretokenized_data_ids.append((tuple(indices), value))
    pretokenized_data = pretokenized_data_ids

    with Pool(processes=num_processes) as p:
        if num_processes == 1:
            pair_counts = [get_pair_counts(value) for value in pretokenized_data]
        else:
            pair_counts = p.imap(get_pair_counts, pretokenized_data, chunksize=100)

        pair_counts_dict = {}
        for pair_count in pair_counts:
            for pair, count in pair_count.items():
                dict_add(pair_counts_dict, pair, count)

        # print(f"{pair_counts_dict=}")

        while len(vocab) < vocab_size:
            max_pair = (None, 0)

            for pair, pair_count in pair_counts_dict.items():
                # print(f"{max_pair=}")
                if pair_count > max_pair[1]:
                    max_pair = (pair, pair_count)
                elif pair_count == max_pair[1] and (vocab[pair[0]], vocab[pair[1]]) > (
                    vocab[max_pair[0][0]],
                    vocab[max_pair[0][1]],
                ):
                    max_pair = (pair, pair_count)
            max_pair = max_pair[0]

            merge_data_func = functools.partial(merge_data, max_pair=max_pair, new_index=index)
            if num_processes == 1:
                merged_data = [merge_data_func(value) for value in pretokenized_data]
            else:
                merged_data = p.imap(
                    merge_data_func,
                    pretokenized_data,
                    chunksize=100,
                )
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            vocab[index] = vocab[max_pair[0]] + vocab[max_pair[1]]
            ivocab[vocab[index]] = index
            index += 1

            pretokenized_data = [None] * len(merged_data)
            for i, (merged, (pairs_to_remove, pairs_to_add)) in enumerate(merged_data):
                pretokenized_data[i] = merged
                for pair, count in pairs_to_remove.items():
                    dict_subtract(pair_counts_dict, pair, count)
                for pair, count in pairs_to_add.items():
                    dict_add(pair_counts_dict, pair, count)
            # print(f"{pair_counts_dict=}")
            # print(f"{pretokenized_data=}")

    return (vocab, merges)


def train_bpe(input_path, vocab_size, special_tokens):
    pretokenized_dict = {}
    special_tokens_bytes = []
    for special_token in special_tokens:
        special_tokens_bytes.append(special_token.encode("utf-8"))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        with Pool(processes=num_processes) as p:
            pretokenized_dicts = p.imap(
                functools.partial(
                    pretokenize_chunk,
                    input_file=f.name,
                    special_tokens=special_tokens,
                    special_tokens_bytes=special_tokens_bytes,
                ),
                zip(boundaries[:-1], boundaries[1:]),
            )

            for token_dict in pretokenized_dicts:
                for token in token_dict:
                    dict_add(pretokenized_dict, token, token_dict[token])

    # print(sorted(pretokenized_dict))
    # print(pretokenized_dict)

    pretokenized_data = pretokenized_dict.items()

    return bpe_on_pretokenized_data(pretokenized_data, vocab_size, special_tokens)


if __name__ == "__main__":
    # train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000000, ["<|endoftext|>"])
    pretokenized_dict = {"and ": 1, " and": 5, "be ": 4, "+": 2, "before": 2}
    bpe_on_pretokenized_data(pretokenized_dict.items(), 10000, [])
