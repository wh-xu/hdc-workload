# %%
import tqdm
import numpy as np


class HDC_GEN:
    def __init__(
        self,
        k: int,
        D: int = 4096,
    ):
        """
        Constructor that creates an array of D-dimensional arrays to store the contents of the hash table
        Creates an encoding scheme for each of the 4 bases: A, C, G, and T

        Parameters:
          k (int): The length of a k-mer. All k-mers in the hash table should have the same length
          D (int): The dimension of a hypervector
        """
        self.k = k
        self.D = D
        self.ref_hvs_table = []

        np.random.seed(0)
        # Generate encoding hvs
        self.encoding_scheme = {}
        for base in ["A", "C", "G", "T"]:
            self.encoding_scheme[base] = (
                np.random.randint(2, size=self.D, dtype=np.int16) * 2 - 1
            )

    def encode(self, kmer: str):
        """
        Encodes a k-mer into a D-dimensional hypervector

        Parameters:
          kmer (str): The kmer we want to encode

        Returns:
          D-dimensional hypervector representing the kmer
        """
        if len(kmer) != self.k:
            raise ValueError(f"k-mer must have length {self.k}")

        enc_hv = np.ones(self.D, dtype=np.int16)
        for i in range(self.k):
            base_enc = self.encoding_scheme[kmer[i]]
            enc_hv *= np.roll(base_enc, i)
        return enc_hv

    def add_ref(self, read: str):
        """
        Adds all the kemrs from a given read to hvs hash table.

        Parameters:
          read (str): The read that we are adding to the hash table
        """
        if len(read) < self.k:
            raise ValueError(f"read must have length >= {self.k}")

        # Use sliding through the read
        kmers = set([read[i : i + self.k] for i in range(len(read) - self.k + 1)])
        ref_hv = np.zeros(self.D, dtype=np.int16)
        for kmer in tqdm.tqdm(kmers, desc="Encoding reference kmers"):
            ref_hv += self.encode(kmer)

        self.ref_hvs_table.append(ref_hv)

    def query(self, kmer: str, threshold: float = 0.8):
        """
        Returns whether or not the given k-mer is in hvs hash table
        If the dot product of the encoded k-mer with any of the HVs is greater than the
        threshold of 0.8 * D, return True. Otherwise, return false.

        Parameters:
          kmer (str): The k-mer we are querying

        Returns:
          Whether or not the k-mer exists in the hash table
        """
        if len(kmer) != self.k:
            raise ValueError(f"read must have length >= {self.k}")

        query_hv = self.encode(kmer)

        largest_dot_prod = 0
        for ref_hv in self.ref_hvs_table:
            dot_prod = ref_hv @ query_hv.T

            if dot_prod > largest_dot_prod:
                largest_dot_prod = dot_prod

            if largest_dot_prod > threshold * self.D:
                break

        return (
            True if largest_dot_prod > threshold * self.D else False
        ), largest_dot_prod


