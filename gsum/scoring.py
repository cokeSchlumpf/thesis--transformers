import pandas as pd
import spacy
import torch

from lib.oracle_summary import extract_oracle_summary
from lib.text_preprocessing import clean_html, preprocess_text, simple_punctuation_only, to_lower
from lib.utils import read_file_to_object, write_object_to_file
from pathlib import Path
from rouge_score import rouge_scorer
from typing import Optional

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


def run_scoring_abs(
        checkpoint_path: str = './data/trained/2021-07-25-2022/gsum-abs-epoch=07-val_loss=2059.89.ckpt',
        reuse_cfg: bool = True,
        max_samples: int = 1000):
    """
    Executes inference with model from a given checkpoint.

    :param checkpoint_path The model/ checkpoint to be used to run inference.
    :param reuse_cfg Reuse config which was used for training.
    :param max_samples The amount of samples to be inferred (might be less, if data loader does not have enough samples)
    """

    checkpoint_path = Path(checkpoint_path)
    scores_path = Path(f'{checkpoint_path.parent}/scores.pkl')

    if reuse_cfg:
        cfg_path = f'{checkpoint_path.parent}/config.json'
        cfg = GuidedSummarizationConfig.from_file(cfg_path)
    else:
        cfg = GuidedSummarizationConfig()

    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum.load_from_checkpoint(str(checkpoint_path), cfg=cfg, data_module=dat)
    run_scoring(mdl, cfg, dat, scores_path, max_samples)


def run_scoring_ext(
        checkpoint_path: str = './data/trained/2021-07-22-0756/gsum-abs-epoch=13-val_loss=133.29.ckpt',
        reuse_cfg: bool = True,
        max_samples: int = 1000):

    checkpoint_path = Path(checkpoint_path)
    scores_path = Path(f'{checkpoint_path.parent}/scores.pkl')

    if reuse_cfg:
        cfg_path = f'{checkpoint_path.parent.absolute()}/config.json'
        cfg = GuidedSummarizationConfig.from_file(cfg_path)
    else:
        cfg = GuidedSummarizationConfig()

    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedExtSum.load_from_checkpoint(str(checkpoint_path), cfg=cfg, data_module=dat)
    run_scoring(mdl, cfg, dat, scores_path, max_samples)


def run_scoring(mdl: torch.nn.Module, cfg: GuidedSummarizationConfig, dat: GuidedSummarizationDataModule, scores_path: Path, max_samples: int = 1000):
    mdl.freeze()
    mdl.to('cuda' if torch.cuda.is_available() else 'cpu')

    dat_loader = dat.inference_dataloader()
    batches_required = int(max_samples / cfg.batch_sizes[3])

    for batch_idx, sample in enumerate(iter(dat_loader)):
        print(f'Processing batch {batch_idx + 1}/{batches_required} ...')

        sources, references = sample

        print('> Create summaries')
        summaries, duration = mdl(sources)
        print('> Created summaries in {:.2f} seconds'.format(duration))

        print('> Calculate rouge scores')
        df_results = pd.DataFrame(data={
            'text': list(sources),
            'summary': list(references),
            'summary_predicted': list(map(lambda s: s.text(), summaries))
        })

        df_results['summary_oracle'] = [extract_oracle_summary(row['text'], row['summary'], dat.lang, summary_length=3)[0] for _, row in df_results.iterrows()]
        df_results = calculate_rouge_scores(df_results, dat.lang)
        df_results = calculate_rouge_scores(df_results, dat.lang, predicted_col='summary_oracle', prefix='oracle_')

        print('> Store results of batch')

        if scores_path.is_file():
            df_existing = read_file_to_object(str(scores_path))
            df_results = pd.concat([df_existing, df_results])

        write_object_to_file(str(scores_path), df_results)
        print(f'> Stored batch, {len(df_results.index)} samples processed now')
        print('✔ Done.')
        print()

        if (batch_idx + 1) >= batches_required:
            break


def calculate_rouge_scores(df: pd.DataFrame, lang: spacy.Language, rouge_n: int = 3, reference_col: str = 'summary', predicted_col: str = 'summary_predicted', prefix: str = '') -> pd.DataFrame:
    """
    Calculates Rouge scores for a set of inferred summaries. The data frame must contain at least two columns:

    - `summary` which includes the reference summaries
    - `summary_predicted` which includes the inferred summary

    The rouge scores are added to the data frame as separate columns.
    """

    result_dict: dict = {}
    rouge_variants = list(map(lambda n: f"rouge{n}", list(range(1, rouge_n + 1)) + ['L']))
    scorer = rouge_scorer.RougeScorer(rouge_variants)
    pipeline = [simple_punctuation_only, to_lower]

    for r in rouge_variants:
        r = r.replace('rouge', 'r')
        result_dict[f'{r}p'] = []
        result_dict[f'{r}r'] = []
        result_dict[f'{r}f'] = []

    for idx, row in df.iterrows():
        row_scores = scorer.score(preprocess_text(row[reference_col], lang, pipeline, [clean_html]), row[predicted_col])

        for r in rouge_variants:
            p = r.replace('rouge', 'r')

            result_dict[f'{p}p'].append(row_scores[r].precision)
            result_dict[f'{p}r'].append(row_scores[r].recall)
            result_dict[f'{p}f'].append(row_scores[r].fmeasure)

    for col in result_dict.keys():
        df[f'{prefix}{col}'] = result_dict[col]

    return df
