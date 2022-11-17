import os
import random
import sys
from logger import SlotAttentionLogger

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from argparse import ArgumentParser

from datasets import CLEVR
from models import QuantizedClassifier
from models import SlotAttentionAE

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--train_path", type=str)

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=2)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--ckpt_test", type=str, default=None)
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--sa_state_dict", type=str, default='./clevr10_sp')
program_parser.add_argument("--pretrained", type=bool, default=False)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--beta", type=float, default=2.)

# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

seed_everything(args.seed, workers=True)

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------


train_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'train'),
                      scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_train_scenes.json'),
                      max_objs=10)

val_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'val'),
                    scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_val_scenes.json'),
                    max_objs=10)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        drop_last=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)

autoencoder = QuantizedClassifier(**dict_args)
if args.pretrained:
    state_dict = torch.load(args.sa_state_dict)
    autoencoder.load_state_dict(state_dict=state_dict, strict=False)

project_name = 'set_prediction_CLEVR'

wandb_logger = WandbLogger(project='set_prediction_CLEVR', name=f'nums {args.nums!r} s {args.seed}', log_model=True)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(save_top_k=1, filename='best', auto_insert_metric_name=False, verbose=True)
every_epoch_callback = ModelCheckpoint(every_n_epochs=10)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# logger_callback= SlotAttentionLogger(val_samples=next(iter(val_dataset)))

callbacks = [
    checkpoint_callback,
    every_epoch_callback,
    # logger_callback,
    # swa,
    # early_stop_callback,
    lr_monitor,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
gpus = [args.gpus]

# trainer
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     #  precision=16,
                     deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)
# Test

trainer.test(dataloaders=val_loader, ckpt_path=None)
