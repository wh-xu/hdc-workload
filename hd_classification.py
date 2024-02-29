# %%
import math
import torch
import tqdm
from hdc import model
from hdc import utils


# %%
# ds_train, ds_test = utils.load_dataset(name="MNIST")
ds_train, ds_test = utils.load_dataset(name="EMG_Hand")
# ds_train, ds_test = utils.load_dataset(name="UCIHAR")
# ds_train, ds_test = utils.load_dataset(name="ISOLET")

n_dim = 2048 # Dimension can be from 1k to 16k
binary = False

n_class = ds_train[1].max() + 1
n_id = ds_train[0].shape[1]


# Data Quantization
if ds_train[0].dtype in [torch.int, torch.uint8, torch.int64]:
    n_lv = int(ds_train[0].max()) + 1
else:
    n_lv = 32
    ds_train = (
        model.min_max_quantize(ds_train[0], int(math.log2(n_lv) - 1)),
        ds_train[1],
    )
    ds_test = (model.min_max_quantize(ds_test[0], int(math.log2(n_lv) - 1)), ds_test[1])


# HDC Model
hdc_model = model.HDC_ID_LV(
    n_class=n_class, n_lv=n_lv, n_id=n_id, n_dim=n_dim, binary=binary
)

# HDC Encoding Step
train_enc = hdc_model.encode(ds_train[0])
test_enc = hdc_model.encode(ds_test[0])

# Init. Training
model.train_init(hdc_model, inp_enc=train_enc, target=ds_train[1])
test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
print(f"Init. test acc. is {test_acc:.4f}")

# Re-training
train_epochs = 20
val_epochs = 5

for i in tqdm.tqdm(range(train_epochs)):
    model.train(hdc_model, inp_enc=train_enc, target=ds_train[1])

    if (i + 1) % val_epochs == 0:
        test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
        print(f"Test acc. @ epoch {i+1}/{train_epochs} is {test_acc:.4f}")

if binary:
    hdc_model.class_hvs = hdc_model.class_hvs.sign()
