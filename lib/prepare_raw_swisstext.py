import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DATA = '../../data/downloaded/swisstext/data_train.csv'
TARGET_DIRECTORY = '../../data/raw/swisstext'


def main(source_data=SOURCE_DATA, target_directory=TARGET_DIRECTORY):
    """
    This program transforms the raw Swisstext data to the input format for all analysis, training and validation
    scripts. See `/data/README.md` for information to obtain the raw data.

    :param source_data: the path to the downloaded training CSV file
    :param target_directory: the path of the target directory for the output files
    :return:
    """
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_data)

    train, test = train_test_split(df, test_size=0.1)
    train, validation = train_test_split(train, test_size=0.1)

    train['source'].to_csv(f"{target_directory}/train.source", index=False, header=['text'])
    train['summary'].to_csv(f"{target_directory}/train.target", index=False, header=['summary'])

    validation['source'].to_csv(f"{target_directory}/validation.source", index=False, header=['text'])
    validation['summary'].to_csv(f"{target_directory}/validation.target", index=False, header=['summary'])

    test['source'].to_csv(f"{target_directory}/test.source", index=False, header=['text'])
    test['summary'].to_csv(f"{target_directory}/test.target", index=False, header=['summary'])

    print(f"Written data files to `{target_directory}`")
    print(f"Training:   {len(train.index)}")
    print(f"Validation: {len(validation.index)}")
    print(f"Test:       {len(test.index)}")


if __name__ == "__main__":
    # execute only if run as a script
    main()
