import multiprocessing as mp
import pytorch_lightning as pl

from datetime import datetime
from lib.utils import write_object_to_file
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


def train_extractive(checkpoint_path: Optional[str] = None):
    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    if checkpoint_path is None:
        mdl = GuidedExtSum(cfg, dat)
    else:
        mdl = GuidedExtSum.load_from_checkpoint(checkpoint_path, cfg=cfg, data_module=dat)

    train(mdl, cfg, dat)


def train_abstractive(checkpoint_path: Optional[str] = None):
    cfg = GuidedSummarizationConfig()
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
    write_object_to_file(f'{training_path}/config.pkl', cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=training_path,
        filename='gsum-abs-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        default_root_dir=training_path,
        val_check_interval=0.25,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback])

    trainer.fit(mdl, dat)
    print('Best model ' + checkpoint_callback.best_model_path)
