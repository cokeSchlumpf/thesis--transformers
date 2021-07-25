import multiprocessing as mp
import os
import pytorch_lightning as pl

from datetime import datetime
from lib.utils import write_string_to_file
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


def train_extractive(checkpoint_path: Optional[str] = None):
    cfg = GuidedSummarizationConfig.apply('mlsum', 'distilbert', True, extractive_preparation_method='oracle')
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    if checkpoint_path is None:
        mdl = GuidedExtSum(cfg, dat)
    else:
        mdl = GuidedExtSum.load_from_checkpoint(checkpoint_path, cfg=cfg, data_module=dat)

    train(mdl, cfg, dat)


def train_abstractive(checkpoint_path: Optional[str] = None):
    cfg = GuidedSummarizationConfig.apply('cnn_dailymail', 'bert', False, guidance_signal='extractive', extractive_preparation_method='oracle', debug=False)
    dat = GuidedSummarizationDataModule(cfg)

    if checkpoint_path is None:
        mdl = GuidedAbsSum(cfg, dat)
    else:
        mdl = GuidedAbsSum.load_from_checkpoint(checkpoint_path, cfg=cfg, data_module=dat)

    train(mdl, cfg, dat)


def train(mdl: pl.LightningModule, cfg: GuidedSummarizationConfig, dat: GuidedSummarizationDataModule) -> None:
    mp.set_start_method('spawn')
    date_time = datetime.now().strftime("%Y-%m-%d-%H%M")

    training_path = f'./data/trained/{date_time}'
    os.mkdir(training_path)
    write_string_to_file(f'{training_path}/config.json', cfg.json())

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=training_path,
        filename='gsum-abs-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        default_root_dir=training_path,
        val_check_interval=1 if cfg.is_debug else 0.1,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback])

    trainer.fit(mdl, dat)
    print('Best model ' + checkpoint_callback.best_model_path)
