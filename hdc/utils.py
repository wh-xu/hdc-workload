# %%
import tqdm
import numpy as np
import torchhd
import torchvision

DATASETS = ["MNIST", "ISOLET", "EMG_Hand", "UCIHAR", "OMS_iPRG_demo"]


def load_dataset(name, path=None):
    """
    Load dataset by name
    """
    if name in DATASETS:
        if name == "MNIST":
            train = torchvision.datasets.MNIST(
                root="./dataset", train=True, download=True
            )
            test = torchvision.datasets.MNIST(
                root="./dataset", train=False, download=True
            )
        elif name == "ISOLET":
            train = torchhd.datasets.ISOLET(root="./dataset", train=True, download=True)
            test = torchhd.datasets.ISOLET(root="./dataset", train=False, download=True)
        elif name == "EMG_Hand":
            data = torchhd.datasets.EMGHandGestures(root="./dataset", download=True)
            n_split = int(len(data) * 0.75)
            train, test = data[:n_split], data[n_split:]
        elif name == "UCIHAR":
            train = torchhd.datasets.UCIHAR(root="./dataset", train=True, download=True)
            test = torchhd.datasets.UCIHAR(root="./dataset", train=False, download=True)
        elif name == "OMS_iPRG_demo":
            train = np.load(path[0])
            test = np.load(path[1])
    else:
        raise NotImplementedError(f"Dataset {name} not implemented!")

    if name in ["EMG_Hand"]:
        data_train, data_test = (train[0].flatten(1).int(), train[1]), (
            test[0].flatten(1).int(),
            test[1],
        )
    elif name in ["MNIST", "ISOLET", "UCIHAR"]:
        data_train, data_test = (train.data.flatten(1), train.targets), (
            test.data.flatten(1),
            test.targets,
        )
    elif name in ["OMS_iPRG_demo"]:

        def convert_csr_to_dense(csr_info, spectra_idx, spectra_intensities):
            """
            convert data in csr format into dense 2d array
            """
            n_spec = len(csr_info) - 1
            n_pts = np.diff(csr_info).max()

            idxs = np.full((n_spec, n_pts), -1, dtype=np.int32)
            levels = np.full((n_spec, n_pts), -1, dtype=np.float32)
            for i in tqdm.tqdm(range(n_spec)):
                i_start, i_end = csr_info[i : i + 2]

                idxs[i][0 : i_end - i_start] = spectra_idx[i_start:i_end]
                levels[i][0 : i_end - i_start] = spectra_intensities[i_start:i_end]

            return idxs, levels

        train_idxs, train_levels = convert_csr_to_dense(
            train["csr_info"], train["spectra_idx"], train["spectra_intensities"]
        )
        data_train = {
            "idxs": train_idxs,
            "levels": train_levels,
            "pr_mzs": train["pr_mzs"],
        }

        test_idxs, test_levels = convert_csr_to_dense(
            test["csr_info"], test["spectra_idx"], test["spectra_intensities"]
        )
        data_test = {
            "idxs": test_idxs,
            "levels": test_levels,
            "pr_mzs": test["pr_mzs"],
        }

    return data_train, data_test

