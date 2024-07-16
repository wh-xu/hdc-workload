# %%
import numba as nb
import numpy as np

import screed
from Bio import SeqIO

import xxhash
import math

import time


def get_seq_list_from_file(fname: str):
    seq_list = []
    seqs = SeqIO.parse(fname, "fasta")
    for record_i in seqs:
        seq_list.append(str(record_i.seq))

    return seq_list


def sample_kmer_list(read: str, k: int, canonical: bool):
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1

    assert num_kmers > 0

    if not canonical:
        kmer_list = [read[i : i + k] for i in range(num_kmers)]
    else:
        kmer_list = [
            min(read[i : i + k], screed.rc(read[i : i + k])) for i in range(num_kmers)
        ]

    return kmer_list


def get_kmer_hashset_from_seq_list(
    seq_list: list, k: int, scaled: int, canonical: bool
):

    scale_factor = np.iinfo(np.uint64).max // scaled

    kmerhash_64 = []
    for i in seq_list:
        kmer_list = sample_kmer_list(i, k, canonical)

        for kmer in kmer_list:
            hash64 = xxhash.xxh64(kmer).intdigest()

            if hash64 <= scale_factor:
                kmerhash_64.append(hash64)

    return set(kmerhash_64)


def encode_hashset_to_hv(kmer_hashset: set, D: int):
    N = len(kmer_hashset)
    D_64 = D // 64

    binary_hvs = np.zeros((N, D), dtype=np.int16)
    for i, h in enumerate(kmer_hashset):
        seed = h

        rnds = []
        for _ in range(D_64):
            seed = xxhash.xxh64(str(h), seed=seed).intdigest()
            rnds.append(seed)

        binary_hv = np.unpackbits(np.array(rnds, dtype=">u8").view(np.uint8)).astype(
            np.int8
        )
        binary_hvs[i, :] = 2 * binary_hv - 1

    return np.sum(binary_hvs, axis=0)


def sketch_fna_to_hv(fname: str, k: int, scaled: int, D: int):
    # Read sequence list from fna file
    start = time.time()
    seq_list = get_seq_list_from_file(fname)
    end = time.time()
    print(f"Reading sequence, Elapsed time: {end-start:.3f}")

    # Sample fractional hashset using scaled factor
    start = time.time()
    kmer_hashset = get_kmer_hashset_from_seq_list(seq_list, k, scaled, True)
    end = time.time()
    print(f"Sampling kmer hashset, Elapsed time: {end-start:.3f}")

    # Encode sampled hashset to create sketch HV
    start = time.time()
    sketch_hv = encode_hashset_to_hv(kmer_hashset, D)
    end = time.time()
    print(f"Sketch HV encoding, Elapsed time: {end-start:.3f}")
    return sketch_hv, kmer_hashset


def Jaccard_HV(hv_A: np.ndarray, hv_B: np.ndarray):
    hv_A, hv_B = hv_A.astype(float), hv_B.astype(float)
    A_B = hv_A @ hv_B.transpose()
    A_2 = hv_A @ hv_A.transpose()
    B_2 = hv_B @ hv_B.transpose()
    jaccard_sim = A_B / (A_2 + B_2 - A_B)
    return jaccard_sim


def Jaccard_set(seq_A, seq_B):
    kmers_A = set(seq_A)
    kmers_B = set(seq_B)

    intersection = kmers_A.intersection(kmers_B)
    union = kmers_A.union(kmers_B)

    sim_score = len(intersection) / len(union)
    return sim_score


def Jaccard_to_ani(jaccard, ksize):
    if jaccard == 0:
        point_estimate = 1.0
    elif jaccard == 1:
        point_estimate = 0.0
    else:
        # point_estimate = 1.0 - (2.0 * jaccard / float(1 + jaccard)) ** (
        #     1.0 / float(ksize)
        # )

        point_estimate = 1.0 + math.log(2 * jaccard / (1 + jaccard)) / float(ksize)
    return point_estimate
