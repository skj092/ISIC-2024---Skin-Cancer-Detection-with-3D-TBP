import numpy as np
import pandas as pd
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import h5py
import io
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Subset
import time

tik = time.time()

kaggle = False

if kaggle:
    root_path = Path('/kaggle/input/isic-2024-challenge')
else:
    root_path = Path('data')
test_image_files = root_path/'test-images/'
test_metadata_path = root_path/'test-metadata.csv'
train_metadata_path = root_path/'train-metadata.csv'
submission_path = 'submission.csv'
test_hdf5 = root_path/'test-image.hdf5'
train_hdf5 = root_path/'train-image.hdf5'

# Use all available GPUs
num_gpus = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = 1 if device == torch.device("cpu") else num_gpus

print(f"Device: {device}")
print(f"Number of GPUs available: {num_gpus}")

model_path = '/kaggle/input/training/model_weights/resnet18.ckpt'
if not kaggle:
    model_path = 'model_weights/resnet18.ckpt'
model_paths = [
    model_path,
]

# create train and validation datasets
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
])


class ISICTestDatast(Dataset):
    def __init__(self, df, test_data, transform=None):
        self.df = df
        self.test_data = test_data
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df.iloc[idx]['isic_id']
        with h5py.File(self.test_data, 'r') as file:
            img = file[id][()]
        img = Image.open(io.BytesIO(img)).convert('RGB')
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return torch.tensor(img).permute(2, 0, 1), id


# load and preprocess data
test_df = pd.read_csv(test_metadata_path)
# define data loader
test_dataset = ISICTestDatast(test_df, test_hdf5, transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=64 * num_gpus,  # Increase batch size proportionally to the number of GPUs
    shuffle=False,
    num_workers=4 * num_gpus,  # Increase number of workers proportionally to the number of GPUs
    pin_memory=True
)

# load and compile models
models = []
for i, model_path in enumerate(model_paths):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_name = 'efficientnet_b0'
    if 'resnet' in model_path:
        model_name = 'resnet18'
    model = timm.create_model(model_name, num_classes=2)
    new_state_dict = {}
    for key, val in state_dict['state_dict'].items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    model.load_state_dict(new_state_dict)
    model.eval()
    model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel
    model.to(device)
    models.append(model)

print(f'{len(models)} models are ready')

# make predictions
ids = []
preds = [np.empty(shape=(0, 2), dtype='float32') for _ in range(len(models))]

# using dataloader
with torch.no_grad():
    for img, filename in tqdm(test_loader):
        img = img.to(device)
        for m_idx in range(len(models)):
            rec_preds = models[m_idx](img).cpu().numpy()
            preds[m_idx] = np.concatenate([preds[m_idx], rec_preds], axis=0)
        ids.extend(filename)

# postprocessing and ensembling
preds = F.sigmoid(torch.Tensor(np.array(preds))).numpy()
s0 = preds.shape[0]
smooth_preds = preds.copy()
preds = smooth_preds.reshape(s0, -1, 2)
preds = preds.mean(axis=0, keepdims=True)
preds = preds.squeeze()

# create submission file
data = pd.read_csv(train_metadata_path)
LABELS = sorted(list(data['target'].unique()))
pred_df = pd.DataFrame(ids, columns=['isic_id'])
pred_df.loc[:, LABELS] = preds
pred_df.to_csv(submission_path, index=False)

# create submission with single label by taking only two columns "isic_id" and "target"
pred_df = pred_df[['isic_id', 1]]
pred_df.columns = ['isic_id', 'target']
pred_df.to_csv(submission_path, index=False)
print(f" top 5 rows of submission file: \n {pred_df.head()}")

tok = time.time()
print("time taken: ", tok-tik)
