# %%
from hdc import hypergen


# Kmer size
k = 21
# Scaled factor to control kmer samlping density
scaled = 2000
# HD dimension
D = 2048


file_query = "./dataset/E.coli/Escherichia_coli_GCA_001440355.LargeContigs.fna"
file_ref = "./dataset/E.coli/Escherichia_coli_IAI39.LargeContigs.fna"

hv_ref, hashset_ref = hypergen.sketch_fna_to_hv(file_ref, k, scaled, D)
hv_query, hashset_query = hypergen.sketch_fna_to_hv(file_query, k, scaled, D)


J_set = hypergen.Jaccard_set(hashset_ref, hashset_query)
J_hv = hypergen.Jaccard_HV(hv_ref, hv_query)
print(f"Ground-truth Jaccard = {J_set:.3f}")
print(f"Jaccard by HyperGen = {J_hv:.3f}")

ANI_set = hypergen.Jaccard_to_ani(J_set, k)
ANI_hv = hypergen.Jaccard_to_ani(J_hv, k)

print(f"ANI similarity by hash set = {ANI_set:.3f}")
print(f"ANI similarity by HyperGen = {ANI_hv:.3f}")
