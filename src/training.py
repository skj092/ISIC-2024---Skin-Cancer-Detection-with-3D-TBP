from training_utils import ISICDatast, ISICDataModule, LitCls
import albumentations as A
import timm
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning import Trainer
from sklearn.model_selection import GroupKFold
import pandas as pd
import json
import warnings
import argparse

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(
    action='ignore', category=pd.errors.SettingWithCopyWarning)

models_path = 'model_weights/'


# from pytorch_lightning.loggers import WandbLogger

# import wandb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-с', '--cfg', help='Model config file path', dest='cfg_path')
    args = {}
    for name, value in vars(parser.parse_args()).items():
        args[name] = value

    train_csv_path = 'data/train-metadata.csv'

    # load config
    with open(args['cfg_path'], 'r') as f:
        CFG = json.load(f)

    # read train metadata
    df_train = pd.read_csv(train_csv_path)

    gkf = GroupKFold(n_splits=5)  # , shuffle=True, random_state=42

    df_train["fold"] = -1
    for idx, (train_idx, val_idx) in enumerate(gkf.split(df_train, df_train["target"], groups=df_train["patient_id"])):
        df_train.loc[val_idx, "fold"] = idx

    # take only one fold
    train_df = df_train[df_train["fold"] != 0].reset_index(drop=True)
    valid_df = df_train[df_train["fold"] == 0].reset_index(drop=True)

    # create train and validation datasets
    transform = A.Compose([
        A.Resize(CFG['img_size'], CFG['img_size']),
        A.Normalize(),
    ])
    train_dataset = ISICDatast(df=train_df,
                               img_dir='data/train-image/image/',
                               transform=transform)
    valid_dataset = ISICDatast(df=valid_df,
                               img_dir='data/train-image/image/',
                               transform=transform)
    datamodule = ISICDataModule(train_dataset, train_dataset,
                                batch_size=CFG['batch_size'])

    # create pretrained model
    model = timm.create_model(
        CFG['model_name'], pretrained=True,
        num_classes=1
    )
    lit_cls = LitCls(model, cutmix_p=0.9, learning_rate=CFG['learning_rate'])

    # create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=None,  # save only last
        filename='{epoch}-{val_AUROC:.3f}',
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    # with open('/wandb_key.txt') as f:
    #     WANDB_KEY = f.readline()
    # wandb.login(key=WANDB_KEY)
    # logger = WandbLogger(
    #     project='BirdCLEF',
    #     log_model=True,
    # )

    # create trainer and start training process
    trainer = Trainer(
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        max_epochs=CFG['epochs_num'],
        accumulate_grad_batches=1,
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        # logger=logger,
        log_every_n_steps=50,
        # accelerator='gpu',
    )
    trainer.fit(lit_cls, datamodule=datamodule)
    # wandb.finish()

    name = args['cfg_path'].split('/')[-1].split('.')[0]
    trainer.save_checkpoint(f'{models_path}{CFG["name"]}.ckpt')
