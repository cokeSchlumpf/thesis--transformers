from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import read_file_to_object

SOURCE_DATA = '../data/downloaded/spon_ard/spon_ard_clean.pkl'
TARGET_DIRECTORY = '../data/raw/spon_ard'


def main(source_data=SOURCE_DATA, target_directory=TARGET_DIRECTORY):
    """
    This program transforms the raw data of zje proprietary collected dataset from Spiegel Online and ARD Teletext
    to the input format for all analysis, training and validation scripts. See `/data/README.md` for information to obtain the raw data.

    :param source_data: the path to the downloaded training CSV file
    :param target_directory: the path of the target directory for the output files
    :return:
    """
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    df = read_file_to_object(source_data)

    df['text_document'] = df['text_document'].apply(lambda s: s.replace('\n', ' '))
    df['text_summary'] = df['text_summary'].apply(lambda s: s.replace('\n', ' '))

    train, test = train_test_split(df, test_size=0.1)
    train, validation = train_test_split(train, test_size=0.1)

    train['text_document'].to_csv(f"{target_directory}/train.source", index=False, header=['text'])
    train['text_summary'].to_csv(f"{target_directory}/train.target", index=False, header=['summary'])

    validation['text_document'].to_csv(f"{target_directory}/validation.source", index=False, header=['text'])
    validation['text_summary'].to_csv(f"{target_directory}/validation.target", index=False, header=['summary'])

    test['text_document'].to_csv(f"{target_directory}/test.source", index=False, header=['text'])
    test['text_summary'].to_csv(f"{target_directory}/test.target", index=False, header=['summary'])

    print(f"Written data files to `{target_directory}`")
    print(f"Training:   {len(train.index)}")
    print(f"Validation: {len(validation.index)}")
    print(f"Test:       {len(test.index)}")


if __name__ == "__main__":
    # execute only if run as a script
    main()
