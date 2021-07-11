import pandas as pd

from lib.utils import read_file_to_object, write_object_to_file
from pathlib import Path
from rouge_score import rouge_scorer

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum


def run_scoring(
        checkpoint_path: str = './data/trained/2021-07-08-0858/gsum-abs-epoch=11-val_loss=3046.30.ckpt',
        max_samples: int = 900):
    """
    Executes inference with model from checkpoint.
    Args:
        checkpoint_path:
        max_samples:

    Returns:

    """

    checkpoint_path = Path(checkpoint_path)
    scores_path = Path(f'{checkpoint_path.parent}/scores.pkl')

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum.load_from_checkpoint(str(checkpoint_path), cfg=cfg, data_module=dat)
    mdl.freeze()

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
        df_results = calculate_rouge_scores(df_results)

        print('> Store results of batch')

        if scores_path.is_file():
            df_existing = read_file_to_object(str(scores_path))
            df_results = pd.concat([df_existing, df_results])

        write_object_to_file(str(scores_path), df_results)
        print(f'> Stored batch, {len(df_results.index)} samples processed now')

        if (batch_idx + 1) > batches_required:
            break


def calculate_rouge_scores(df: pd.DataFrame, rouge_n: int = 3) -> pd.DataFrame:
    result_dict: dict = {}
    rouge_variants = list(map(lambda n: f"rouge{n}", list(range(1, rouge_n + 1)) + ['L']))
    scorer = rouge_scorer.RougeScorer(rouge_variants)

    for r in rouge_variants:
        r = r.replace('rouge', 'r')
        result_dict[f'{r}p'] = []
        result_dict[f'{r}r'] = []
        result_dict[f'{r}f'] = []

    for idx, row in df.iterrows():
        row_scores = scorer.score(row['summary'], row['summary_predicted'])  # TODO: Add preprocessing to have similar text quality?

        for r in rouge_variants:
            p = r.replace('rouge', 'r')

            result_dict[f'{p}p'].append(row_scores[r].precision)
            result_dict[f'{p}r'].append(row_scores[r].recall)
            result_dict[f'{p}f'].append(row_scores[r].fmeasure)

    for col in result_dict.keys():
        df[col] = result_dict[col]

    return df
