import multiprocessing as mp
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .scoring import run_scoring
from .summarizer import BeamSearchResult, GuidedAbsSum, GuidedExtSum


def run_extractive():
    mp.set_start_method('spawn')
    
    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    mdl = GuidedExtSum(cfg, dat)

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        max_epochs=5,
        default_root_dir='./data/checkpoints')

    trainer.fit(mdl, dat)
    trainer.test(mdl, dat.test_dataloader())


def run_abstractive():
    mp.set_start_method('spawn')
    date_time = datetime.now().strftime("%Y-%m-%d-%H%M")

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum(cfg, dat)

    training_path = f'./data/trained/{date_time}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=training_path,
        filename='gsum-abs-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        default_root_dir=training_path,
        val_check_interval=0.25,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback])

    trainer.fit(mdl, dat)
    print('Best model ' + checkpoint_callback.best_model_path)
    trainer.test(mdl, dat.test_dataloader())


def run_test():
    run_scoring()
