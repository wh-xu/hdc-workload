# %%
import math
import torch
import torchhd

import tqdm

torch.manual_seed(0)


def min_max_quantize(inp, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(inp) - 1
    min_val, max_val = inp.min(), inp.max()

    input_rescale = (inp - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = v * (max_val - min_val) + min_val
    v = torch.round(v * n)
    v = v - v.min()
    return v.int()


def train_init(model, inp_enc, target, binary=True):
    assert inp_enc.shape[0] == target.shape[0]

    for i in range(model.n_class):
        idx = target == i
        model.class_hvs[i] = (
            inp_enc[idx].sum(dim=0).sign() if binary else inp_enc[idx].sum(dim=0)
        )


def test(model, inp_enc, target, binary=True):
    assert inp_enc.shape[0] == target.shape[0]

    # Distance matching
    if binary:
        dist = torch.matmul(inp_enc, model.class_hvs.sign().T)
    else:
        dist = torch.matmul(inp_enc, model.class_hvs.T)
        dist = dist / model.class_hvs.float().norm(dim=1)

    acc = dist.argmax(dim=1) == target.long()
    acc = acc.float().mean()

    return acc


def train(model, inp_enc, target):
    assert inp_enc.shape[0] == target.shape[0]

    n_samples = inp_enc.shape[0]

    for j in range(n_samples):
        pred = torch.matmul(inp_enc[j], model.class_hvs.T).argmax()

        if pred != target[j]:
            model.class_hvs[target[j]] += inp_enc[j]
            model.class_hvs[pred] -= inp_enc[j]


class HDC_ID_LV:
    def __init__(self, n_class, n_lv, n_id, n_dim, binary=True) -> None:
        self.n_class = n_class
        self.n_dim, self.binary = n_dim, binary
        self.n_lv, self.n_id = n_lv, n_id
        self.hv_lv = torch.randint(0, 2, size=(n_lv, n_dim), dtype=torch.int) * 2 - 1
        self.hv_id = torch.randint(0, 2, size=(n_id, n_dim), dtype=torch.int) * 2 - 1

        self.class_hvs = torch.zeros(n_class, n_dim, dtype=torch.int)

    def encode(self, inp):
        assert inp.shape[1] == self.n_id

        # ID-LV encoding
        n_batch = inp.shape[0]
        inp_enc = torch.zeros(n_batch, self.n_dim, dtype=torch.int)
        for i in tqdm.tqdm(range(n_batch)):
            # Vectorized version
            # print(self.hv_lv.shape)
            # print(self.hv_lv[inp[i].long()].shape)
            inp_enc[i] = (self.hv_id * self.hv_lv[inp[i].long()]).sum(dim=0)

            # Serial version
            # tmp = torch.zeros(1, self.n_dim, dtype=torch.int)
            # for j in range(self.n_id):
            # tmp = tmp + (self.hv_id[j] * self.hv_lv[inp_quant[i][j]])
            # inp_enc[i] = tmp

        return inp_enc.sign().int() if self.binary else inp_enc


class HDC_RP:
    def __init__(self, n_class, n_feat, n_dim, binary=True) -> None:
        self.n_class = n_class
        self.n_dim, self.n_feat = n_dim, n_feat
        self.binary = binary
        self.rp = torch.randint(0, 2, size=(n_feat, n_dim), dtype=torch.int) * 2 - 1
        self.class_hvs = torch.zeros(n_class, n_dim, dtype=torch.int)

    def encode(self, inp):
        assert inp.shape[1] == self.n_feat

        inp_enc = torch.matmul(inp, self.rp)
        return inp_enc.sign().int() if self.binary else inp_enc
