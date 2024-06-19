# %%
from Bio import SeqIO
from hdc import genome

# %% Load Reference Genome Dataset
genome_file = "./dataset/E.coli_str_K-12_substr_MG1655.fna"
for record in SeqIO.parse(genome_file, "fasta"):
    ref = str(record.seq)

# %% HDC Model
k = 200  # genome read length
n_dim = 8192  # HV Dimension can be from 1k to 16k
hd_db = genome.HDC_GEN(k=k, D=n_dim)

# Create reference hvs
hd_db.add_ref(ref)


# %% Query existing genome seq.
Q = ref[0:k]  # check if query exists in the reference

if_exist, similarity = hd_db.query(Q)
print(f"If exist={if_exist}, sim={similarity:.3f}")


# %% Query non-existing genome seq.
Q = list(Q)
Q[5:8] = "CCC"  # Modify partial data
Q = "".join(Q)

if_exist, similarity = hd_db.query(Q)
print(f"If exist={if_exist}, sim={similarity:.3f}")
