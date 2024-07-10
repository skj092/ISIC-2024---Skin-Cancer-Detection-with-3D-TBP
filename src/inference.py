import code
import sys
import numpy as np
import pandas as pd
import os
import joblib
import openvino as ov
import librosa
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A


def torch_to_ov(model, input_shape=[48, 1, 128, 0]):
    core = ov.Core()
    ov_model = ov.convert_model(model)
    # ov_model.reshape(input_shape)
    compiled_model = core.compile_model(ov_model)
    return compiled_model


model_paths = [
    # EfficientNet_b0
    'model_weights/resnet18.ckpt',
    'model_weights/resnet18.ckpt',
]

test_image_files = 'tmp/test-images/'
test_metadat_path = 'data/test-metadata.csv'
train_metadata_path = 'data/train-metadata.csv'
submission_path = 'tmp/submission.csv'
device = "cuda" if torch.cuda.is_available() else "cpu"

# create train and validation datasets
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
])


class ISICTestDatast(Dataset):
    def __init__(self,
                 df,
                 img_dir,
                 transform=None,
                 **kwargs):
        df = df.reset_index(drop=True)
        self.transform = transform
        self.df = df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'isic_id']
        img_path = os.path.join(self.img_dir, img_id + '.png')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        label = torch.tensor([0, 1], dtype=torch.float32)
        return torch.tensor(img).permute(2, 0, 1), label, img_id


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
    model.to(device)
    # model = torch_to_ov(model, input_shape=SHAPE)
    models.append(model)
print(f'{len(models)} models are ready')

# load and preprocess data
test_df = pd.read_csv(test_metadat_path)
test_df['img_path'] = test_df['isic_id'].apply(
    lambda x: os.path.join(test_image_files, f'{x}.png'))

# define data loader
test_dataset = ISICTestDatast(test_df, test_image_files, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def process(idx):
    row = test_df.iloc[idx]
    audiopath = row['img_path']
    img_arr = np.array(Image.open(audiopath))
    img_arr = img_arr / 255.0
    img_arr = img_arr.transpose(2, 0, 1)
    return row['isic_id'], img_arr


#indexes = test_df.index
#output = joblib.Parallel(n_jobs=-1, backend="loky")(
#    joblib.delayed(process)(idx) for idx in indexes
#)
#filenames, image_array = zip(*output)

# make predictions
ids = []
preds = [np.empty(shape=(0, 2), dtype='float32') for _ in range(len(models))]
#with torch.no_grad():
#    for filename, img in zip(filenames, image_array):
#        image = torch.Tensor(img).unsqueeze(0)
#        for m_idx in range(len(models)):
#            rec_preds = models[m_idx](image).numpy()
#            preds[m_idx] = np.concatenate([preds[m_idx], rec_preds], axis=0)
#        ids.append(filename)

# using dataloader
with torch.no_grad():
    for img, label, filename in test_loader:
        img = img.to(device)
        for m_idx in range(len(models)):
            rec_preds = models[m_idx](img).cpu().numpy()
            preds[m_idx] = np.concatenate([preds[m_idx], rec_preds], axis=0)
        ids.extend(filename)

# postprocessing and ensembling
preds = F.sigmoid(torch.Tensor(np.array(preds))).numpy()
s0 = preds.shape[0]
# preds = preds.reshape(s0, -1, 48, 182)
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
