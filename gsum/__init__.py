from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .scoring import run_scoring
from .summarizer import BeamSearchResult, GuidedAbsSum, GuidedExtSum
from .train import train_abstractive, train_extractive


def run_extractive():
    train_extractive()


def run_abstractive():
    train_abstractive()


def run_test():
    run_scoring()
