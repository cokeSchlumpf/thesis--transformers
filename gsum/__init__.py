import multiprocessing as mp
import pytorch_lightning as pl

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


def run_extractive():
    mp.set_start_method('spawn')
    
    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    mdl = GuidedExtSum(cfg, dat)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        default_root_dir='./data/checkpoints')

    trainer.fit(mdl, dat)
    trainer.test(mdl, dat.test_dataloader())


def run_abstractive():
    mp.set_start_method('spawn')

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum(cfg, dat)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=5,
        default_root_dir='./data/checkpoints')

    trainer.fit(mdl, dat)
    trainer.test(mdl, dat.test_dataloader())