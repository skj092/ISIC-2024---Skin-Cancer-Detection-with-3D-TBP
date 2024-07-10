from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics.classification import MultilabelAUROC
from torchmetrics import MetricCollection, MeanMetric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
import warnings
import os
from PIL import Image

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(
    action='ignore', category=pd.errors.SettingWithCopyWarning)


class ISICDatast(Dataset):
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
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'target'] # 0 or 1
        label = torch.tensor([label, 1-label], dtype=torch.float32)
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return img, label


class ISICDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=False,
                          shuffle=True,
                          )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=False,
                          shuffle=False,
                          )


class CutMix:
    def __init__(self,
                 mode: str = 'horizontal',
                 p: float = 1.0,
                 cuts_num: int = 1):
        assert mode in ['horizontal']
        self.mode = mode
        self.cuts_num = cuts_num
        self.p = p

    def apply_horizontal(self, imgs, labels):
        w = imgs.shape[-1]
        b = imgs.shape[0]

        alphas = np.sort(np.random.rand(self.cuts_num))
        rand_index = [np.random.permutation(b) for _ in range(self.cuts_num)]
        imgs_tomix = [imgs[idxes] for idxes in rand_index]
        labels_tomix = [labels[idxes] for idxes in rand_index]

        for alpha, img_tomix in zip(alphas, imgs_tomix):
            imgs[..., int(alpha*w):] = img_tomix[..., int(alpha*w):]

        labels = labels*alphas[0]
        for i in range(1, self.cuts_num):
            labels += labels_tomix[i-1]*(alphas[i] - alphas[i-1])
        labels += labels_tomix[-1] * (1 - alphas[-1])

        return imgs, labels

    def __call__(self, imgs, labels):
        if random.random() > self.p:
            return imgs, labels
        if self.mode in ['horizontal']:
            imgs, labels = self.apply_horizontal(imgs, labels)
        return imgs, labels


class LitCls(LightningModule):

    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = 3e-4,
            cutmix_p: float = 0,
            cuts_num: int = 1,
    ) -> None:
        super().__init__()

        self.model: torch.nn.Module = model
        self.learning_rate: float = learning_rate
        self.aug_cutmix = CutMix(
            mode='horizontal', p=cutmix_p, cuts_num=cuts_num)

        self.loss: torch.nn.Module = nn.CrossEntropyLoss()
        metric_ce = MetricCollection({
            "CE": MeanMetric()
        })
        metric_auroc = MetricCollection({
            "AUROC": MultilabelAUROC(num_labels=2, average="macro"),
        })

        self.train_ce: MetricCollection = metric_ce.clone(prefix="train_")
        self.val_ce: MetricCollection = metric_ce.clone(prefix="val_")
        self.train_auroc: MetricCollection = metric_auroc.clone(
            prefix="train_")
        self.val_auroc: MetricCollection = metric_auroc.clone(prefix="val_")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x = x.transpose(1, 3) # (B, H, W, C) -> (B, C, H, W)
        x, y = self.aug_cutmix(x, y)
        preds = self.model(x)
        train_loss = self.loss(preds, y)
        self.train_ce(train_loss)
        self.log('train_loss', train_loss, prog_bar=True, sync_dist=True)
        self.train_auroc(F.sigmoid(preds), (y+0.9).int())
        return train_loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_ce.compute(), sync_dist=True)
        self.train_ce.reset()
        self.log_dict(self.train_auroc.compute(), sync_dist=True)
        self.train_auroc.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x = x.transpose(1, 3) # (B, H, W, C) -> (B, C, H, W)
        preds = self.model(x)
        val_loss = self.loss(preds, y)
        self.val_ce(val_loss)
        self.val_auroc(F.sigmoid(preds), (y+0.9).int())

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ce.compute(), prog_bar=True, sync_dist=True)
        self.val_ce.reset()
        self.log_dict(self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.trainer.model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
