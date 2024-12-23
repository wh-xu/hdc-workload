# %%
import os
import numpy as np
from hdc import utils

# %% Load Mass Spec. Dataset
# ref_dataset = "iprg"  # benchmark the small iPRG2012 dataset
ref_dataset = "hcd"  # benchmark the large Massive Human HCD dataset

charge_start, charge_end = 1, 9


dict_charge_spectra = {}
for charge in range(charge_start, charge_end + 1):
    if ref_dataset == "iprg":
        dim_spectra = 34976
        ref_fname = f"./dataset/oms/ref/human_yeast_targetdecoy_vec_{dim_spectra}.charge{charge}.npz"
        query_fname = (
            f"./dataset/oms/query/iPRG2012_vec_{dim_spectra}.charge{charge}.npz"
        )
    elif ref_dataset == "hcd":
        dim_spectra = 27981
        query_idx = 0  # pick one of the query files to test
        ref_fname = f"./dataset/oms/ref/massive_human_hcd_unique_targetdecoy_vec_{dim_spectra}.charge{charge}.npz"
        query_fname = f"./dataset/oms/query/{utils.hdc_query_list[query_idx]}_vec_{dim_spectra}.charge{charge}.npz"
    else:
        raise NotImplementedError("Dataset not implemented")

    if not os.path.exists(query_fname) or not os.path.exists(ref_fname):
        continue

    ds_ref, ds_query = utils.load_dataset(
        name="OMS_iPRG_demo", path=[ref_fname, query_fname]
    )
    n_ref, n_query = len(ds_ref["pr_mzs"]), len(ds_query["pr_mzs"])

    print(f"Charge-{charge}\t{n_ref} references and {n_query} queries loaded")

    dict_charge_spectra[charge] = (n_ref, n_query)


print(
    f"###############################################################################"
)
n_ref_total = np.sum([i[0] for i in dict_charge_spectra.values()])
n_query_total = np.sum([i[1] for i in dict_charge_spectra.values()])
print(
    f"Total spectra: {n_ref_total} references and {n_query_total} queries for charge {charge_start} to {charge_end}"
)
print(
    f"###############################################################################"
)
