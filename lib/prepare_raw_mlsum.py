import pandas as pd

from pathlib import Path
from datasets import load_dataset

TARGET_DIRECTORY = '../data/raw/mlsum'


def main(target_directory=TARGET_DIRECTORY):
    """
    This program transforms the raw MLSum Dataset data provided by huggingface's data library into the
    required format for this project.

    :param target_directory: the path of the target directory for the output files
    :return:
    """

    def transform_dataset(ds):
        return {
            'text': [text.replace('\n', ' ') for text in ds['text']],
            'summary': [summary.replace('\n', ' ') for summary in ds['summary']]
        }

    Path(target_directory).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mlsum", "de")
    train = pd.DataFrame(data=transform_dataset(dataset['train']))
    test = pd.DataFrame(data=transform_dataset(dataset['test']))
    validation = pd.DataFrame(data=transform_dataset(dataset['validation']))

    train['text'].to_csv(f"{target_directory}/train.source", index=False, header=['text'])
    train['summary'].to_csv(f"{target_directory}/train.target", index=False, header=['summary'])

    validation['text'].to_csv(f"{target_directory}/validation.source", index=False, header=['text'])
    validation['summary'].to_csv(f"{target_directory}/validation.target", index=False, header=['summary'])

    test['text'].to_csv(f"{target_directory}/test.source", index=False, header=['text'])
    test['summary'].to_csv(f"{target_directory}/test.target", index=False, header=['summary'])

    print(f"Written data files to `{target_directory}`")
    print(f"Training:   {len(train.index)}")
    print(f"Validation: {len(validation.index)}")
    print(f"Test:       {len(test.index)}")


if __name__ == "__main__":
    # execute only if run as a script
    main()
