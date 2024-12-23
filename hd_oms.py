# %%
import math
import torch
import tqdm
from hdc import model
from hdc import utils


# %% Load Mass Spec. Dataset
charge = 2

# ref_dataset = "iprg"  # benchmark the small iPRG2012 dataset
ref_dataset = "hcd"  # benchmark the large Massive Human HCD dataset

# Just pick up a subset for quick evaluation
n_ref_test = 1000
n_query_test = 10
# topk results to search
topk = 5

if ref_dataset == "iprg":
    dim_spectra = 34976
    ref_fname = (
        f"./dataset/human_yeast_targetdecoy_vec_{dim_spectra}.charge{charge}.npz"
    )
    query_fname = f"./dataset/iPRG2012_vec_{dim_spectra}.charge{charge}.npz"
elif ref_dataset == "hcd":
    dim_spectra = 27981
    query_idx = 0  # pick one of the query files to test
    ref_fname = f"./dataset/oms/ref/massive_human_hcd_unique_targetdecoy_vec_{dim_spectra}.charge{charge}.npz"
    query_fname = f"./dataset/oms/query/{utils.hdc_query_list[query_idx]}_vec_{dim_spectra}.charge{charge}.npz"
else:
    raise NotImplementedError("Dataset not implemented")

ds_ref, ds_query = utils.load_dataset(
    name="OMS_iPRG_demo", path=[ref_fname, query_fname]
)

n_ref, n_query = len(ds_ref["pr_mzs"]), len(ds_query["pr_mzs"])

n_id = dim_spectra  # Num of id HVs
n_lv = 64  # Quantization levels for level HVs
n_dim = 2048  # HV Dimension can be from 1k to 16k
binary = True

# HDC Model
hdc_model = model.HDC_ID_LV(
    n_class=n_ref,
    n_lv=n_lv,
    n_id=n_id,
    n_dim=n_dim,
    method_id_lv="cyclic",
    binary=binary,
)

# Data Quantization to n_lv levels
ds_ref["levels"][ds_ref["levels"] == -1] = 0
ds_ref_levels_quantized = model.min_max_quantize(
    torch.tensor(ds_ref["levels"]), int(math.log2(n_lv) - 1)
)
ds_ref_idxs = torch.tensor(ds_ref["idxs"])

ds_query["levels"][ds_query["levels"] == -1] = 0
ds_query_levels_quantized = model.min_max_quantize(
    torch.tensor(ds_query["levels"]), int(math.log2(n_lv) - 1)
)
ds_query_idxs = torch.tensor(ds_query["idxs"])


# %% HDC Encoding Step for Database Pre-building
ref_enc = hdc_model.encode(
    {"lv": ds_ref_levels_quantized[:n_ref_test], "idx": ds_ref_idxs[:n_ref_test]},
    dense=False,
)

# %% HDC Encoding Step for Querying
query_enc = hdc_model.encode(
    {
        "lv": ds_query_levels_quantized[:n_query_test],
        "idx": ds_query_idxs[:n_query_test],
    },
    dense=False,
)

# %% Database Search
ip = torch.matmul(query_enc, ref_enc.T)
if not hdc_model.binary:
    dist = ip / ref_enc.float().norm(dim=1)
sim, pred = torch.topk(ip, topk, dim=-1)

print(
    f"###############################################################################"
)
print(
    f"{n_ref_test} of {n_ref} references and {n_query_test} of {n_query} queries are used for testing"
)
print("Topk index results:\n", pred, "\nwith similarity\n", sim)
print(
    f"###############################################################################"
)
