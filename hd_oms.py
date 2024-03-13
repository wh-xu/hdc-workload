# %%
import math
import torch
import tqdm
from hdc import model
from hdc import utils


# %% Load Mass Spec. Dataset
dim_spectra = 34976
charge = 2

ref_fname = f"./dataset/human_yeast_targetdecoy_vec_{dim_spectra}.charge{charge}.npz"
query_fname = f"./dataset/iPRG2012_vec_{dim_spectra}.charge{charge}.npz"

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
n_test = 100
ref_enc = hdc_model.encode(
    {"lv": ds_ref_levels_quantized[:n_test], "idx": ds_ref_idxs[:n_test]}, dense=False
)  # Just pick up a few samples for quick evaluation

# %% HDC Encoding Step for Querying
n_query = 5
query_enc = hdc_model.encode(
    {"lv": ds_query_levels_quantized[:n_test], "idx": ds_query_idxs[:n_test]},
    dense=False,
)  # Just pick up a few samples for quick evaluation


# %% Database Search
dist = torch.matmul(query_enc, ref_enc.T)
if hdc_model.binary:
    pred = dist.argmax(dim=-1)
else:
    dist = dist / ref_enc.float().norm(dim=1)
    pred = dist.argmax(dim=-1)
