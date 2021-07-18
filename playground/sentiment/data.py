import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import spacy
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm

from lib.text_preprocessing import preprocess_text, clean_html, to_lower, simple_punctuation_only, lemmatize, remove_stopwords
from lib.utils import checksum_of_file, read_file_to_object, read_file_to_string, write_object_to_file, write_string_to_file

tqdm.pandas()


class SentimentDataset(Dataset):

    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        result = [torch.from_numpy(self.x[idx].toarray()).type(torch.FloatTensor)[0], torch.tensor(self.y[idx])]
        return result


class SentimentDataModule(pl.LightningDataModule):

    def __init__(self,
                 raw_path: str = './data/raw/IMDB Dataset.csv',
                 prepared_dir_path: str = './data/prepared',
                 spacy_model: str = 'en_core_web_sm',
                 train_test_validate_split: (float, float, float) = (0.7, 0.2, 0.1),
                 vector_dimensions: int = 50000,
                 batch_sizes: (int, int, int) = (20, 20, 20)):

        super().__init__()
        self.raw_path = raw_path
        self.prepared_dir_path = prepared_dir_path
        self.transform_pipeline = [lemmatize, remove_stopwords, simple_punctuation_only, to_lower]
        self.transform_lang = spacy.load(spacy_model)
        self.train_test_validate_split = train_test_validate_split
        self.vector_dimensions = vector_dimensions
        self.batch_sizes = batch_sizes

        self.train_raw_path = self.prepared_dir_path + '/sentiment.train.raw.csv'
        self.test_raw_path = self.prepared_dir_path + '/sentiment.test.raw.csv'
        self.validate_raw_path = self.prepared_dir_path + '/sentiment.validate.raw.csv'

        self.train_raw_checksum_path = self.prepared_dir_path + '/sentiment.train.raw.csv.md5'
        self.test_raw_checksum_path = self.prepared_dir_path + '/sentiment.test.raw.csv.md5'
        self.validate_raw_checksum_path = self.prepared_dir_path + '/sentiment.validate.raw.csv.md5'

        self.train_cleaned_path = self.prepared_dir_path + '/sentiment.train.cleaned.csv'
        self.test_cleaned_path = self.prepared_dir_path + '/sentiment.test.cleaned.csv'
        self.validate_cleaned_path = self.prepared_dir_path + '/sentiment.validate.cleaned.csv'

        self.train_cleaned_checksum_path = self.prepared_dir_path + '/sentiment.train.cleaned.csv.md5'
        self.test_cleaned_checksum_path = self.prepared_dir_path + '/sentiment.test.cleaned.csv.md5'
        self.validate_cleaned_checksum_path = self.prepared_dir_path + '/sentiment.validate.cleaned.csv.md5'

        self.train_prepared_path = self.prepared_dir_path + '/sentiment.train_prepared.pkl'
        self.test_prepared_path = self.prepared_dir_path + '/sentiment.test_prepared.pkl'
        self.validate_prepared_path = self.prepared_dir_path + '/sentiment.validate_prepared.pkl'

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vectorizer_path = self.prepared_dir_path + '/sentiment.vectorizer.pkl'
        self.vectorizer_checksum_path = self.prepared_dir_path + '/sentiment.vectorizer.pkl.md5'
        self.encoder: Optional[LabelEncoder] = None
        self.encoder_path = self.prepared_dir_path + '/sentiment.encoder.pkl'
        self.encoder_checksum_path = self.prepared_dir_path + '/sentiment.encoder.pkl.md5'

        self.train: Optional[SentimentDataset] = None
        self.test: Optional[SentimentDataset] = None
        self.validate: Optional[SentimentDataset] = None

    def prepare_data(self) -> None:
        #
        # Split raw dataset into train, test and validation sets.
        #
        if (os.path.isfile(self.train_raw_path) is False) or (os.path.isfile(self.test_raw_path) is False) or (os.path.isfile(self.validate_raw_path) is False):
            df = pd.read_csv(self.raw_path)
            train, test = train_test_split(df, train_size=self.train_test_validate_split[0])
            test, validate = train_test_split(test, train_size=(self.train_test_validate_split[1] / (self.train_test_validate_split[1] + self.train_test_validate_split[2])))

            train.to_csv(self.train_raw_path)
            test.to_csv(self.test_raw_path)
            validate.to_csv(self.validate_raw_path)

        #
        # Prepare input data (clean text)
        #
        self.__clean_dataset(self.train_raw_path, self.train_raw_checksum_path, self.train_cleaned_path)
        self.__clean_dataset(self.test_raw_path, self.test_raw_checksum_path, self.test_cleaned_path)
        self.__clean_dataset(self.validate_raw_path, self.validate_raw_checksum_path, self.validate_cleaned_path)

        #
        # Update checksum of raw data files
        #
        write_string_to_file(self.train_raw_checksum_path, checksum_of_file(self.train_raw_path))
        write_string_to_file(self.test_raw_checksum_path, checksum_of_file(self.test_raw_path))
        write_string_to_file(self.validate_raw_checksum_path, checksum_of_file(self.validate_raw_path))

        #
        # Prepare tokenizer and encoder
        #
        train_cleaned_checksum = checksum_of_file(self.train_cleaned_path)
        train_cleaned_checksum_exists = read_file_to_string(self.train_cleaned_checksum_path)

        if (train_cleaned_checksum != train_cleaned_checksum_exists) or (os.path.isfile(self.vectorizer_path) is False):
            df = pd.read_csv(self.train_cleaned_path)
            self.vectorizer = TfidfVectorizer(max_features=self.vector_dimensions)
            self.vectorizer.fit(df['review'])

            write_object_to_file(self.vectorizer_path, self.vectorizer)
        else:
            self.vectorizer = read_file_to_object(self.vectorizer_path)

        if (train_cleaned_checksum != train_cleaned_checksum_exists) or (os.path.isfile(self.encoder_path) is False):
            df = pd.read_csv(self.train_cleaned_path)

            self.encoder = LabelEncoder()
            self.encoder.fit(df['sentiment'].to_numpy())

            write_object_to_file(self.encoder_path, self.encoder)
        else:
            self.encoder = read_file_to_object(self.encoder_path)

        #
        # Update checksum of cleaned data files
        #
        write_string_to_file(self.train_cleaned_checksum_path, checksum_of_file(self.train_cleaned_path))
        write_string_to_file(self.test_cleaned_checksum_path, checksum_of_file(self.test_cleaned_path))
        write_string_to_file(self.validate_cleaned_checksum_path, checksum_of_file(self.validate_cleaned_path))

        #
        # Prepare data
        #
        vectorizer_checksum = checksum_of_file(self.vectorizer_path)
        vectorizer_checksum_exists = read_file_to_string(self.vectorizer_checksum_path)
        vectorizer_changed = vectorizer_checksum != vectorizer_checksum_exists

        encoder_checksum = checksum_of_file(self.encoder_path)
        encoder_checksum_exists = read_file_to_string(self.encoder_checksum_path)
        encoder_changed = encoder_checksum != encoder_checksum_exists

        if encoder_changed or vectorizer_changed or (os.path.isfile(self.train_prepared_path) is False):
            self.__prepare_dataset(self.train_cleaned_path, self.train_prepared_path)

        if encoder_changed or vectorizer_changed or (os.path.isfile(self.test_prepared_path) is False):
            self.__prepare_dataset(self.test_cleaned_path, self.test_prepared_path)

        if encoder_changed or vectorizer_changed or (os.path.isfile(self.validate_prepared_path) is False):
            self.__prepare_dataset(self.validate_cleaned_path, self.validate_prepared_path)

        #
        # Update checksums of vectorizer and encoder
        #
        write_string_to_file(self.vectorizer_checksum_path, vectorizer_checksum)
        write_string_to_file(self.encoder_checksum_path, encoder_checksum)

    def __clean_dataset(self, input_file: str, input_checksum_file: str, output_file: str) -> None:
        input_checksum = checksum_of_file(input_file)
        input_checksum_exists = read_file_to_string(input_checksum_file)

        if (input_checksum != input_checksum_exists) or (os.path.isfile(output_file) is False):
            df = pd.read_csv(input_file)
            df['review'] = df['review'].progress_apply(lambda s: preprocess_text(s, self.transform_lang, self.transform_pipeline, [clean_html]))
            df.to_csv(output_file)

    def __prepare_dataset(self, input_file: str, output_file: str) -> None:
        print(f"Preparing ${output_file} ...")
        df = pd.read_csv(input_file)
        x = self.vectorizer.transform(df['review'])
        y = self.encoder.transform(df['sentiment'])
        dataset = SentimentDataset(x, y)
        write_object_to_file(output_file, dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        self.vectorizer = read_file_to_object(self.vectorizer_path)
        self.encoder = read_file_to_object(self.encoder_path)
        self.train = read_file_to_object(self.train_prepared_path)
        self.test = read_file_to_object(self.test_prepared_path)
        self.validate = read_file_to_object(self.validate_prepared_path)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if self.train is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            return DataLoader(self.train, self.batch_sizes[0], num_workers=16)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.test is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            result = DataLoader(self.test, self.batch_sizes[1], num_workers=16)
            return result

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.validate is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            return DataLoader(self.validate, self.batch_sizes[2], num_workers=16)

    def transfer_batch_to_device(self, batch: any, device: Optional[torch.device] = None) -> any:
        return [batch[0].to(device), batch[1].to(device)]

    def preprocess(self, x: List[str], device: torch.device) -> torch.Tensor:
        cleaned = [preprocess_text(s, self.transform_lang, self.transform_pipeline, [clean_html]) for s in x]
        vectorized = self.vectorizer.transform(cleaned)
        return torch.from_numpy(vectorized.toarray()).type(torch.FloatTensor).to(device)