import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import spacy
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm

from lib.text_preprocessing import preprocess_text, clean_html, to_lower, simple_punctuation_only, lemmatize, remove_stopwords
from lib.utils import checksum_of_file, read_file_to_object, read_file_to_string, write_object_to_file, write_string_to_file

tqdm.pandas()


class SentimentDataset(Dataset):

    def __init__(self, x: dict, y: np.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x['input_ids'].shape[0]

    def __getitem__(self, idx):
        result = [
            {
                'input_ids': self.x['input_ids'][idx],
                'token_type_ids': self.x['token_type_ids'][idx],
                'attention_mask': self.x['attention_mask'][idx]
            },
            torch.tensor(self.y[idx])
        ]

        return result


class SentimentDataModule(pl.LightningDataModule):

    def __init__(self,
                 raw_path: str = './data/raw/IMDB Dataset.csv',
                 prepared_dir_path: str = './data/prepared',
                 spacy_model: str = 'en_core_web_sm',
                 transformer_model: str = 'bert-base-uncased',
                 train_test_validate_split: (float, float, float) = (0.7, 0.2, 0.1),
                 batch_sizes: (int, int, int) = (200, 200, 200)):

        super().__init__()
        self.raw_path = raw_path
        self.prepared_dir_path = prepared_dir_path
        self.transform_pipeline = [lemmatize, remove_stopwords, simple_punctuation_only, to_lower]
        self.transform_lang = spacy.load(spacy_model)
        self.train_test_validate_split = train_test_validate_split
        self.transformer_model = transformer_model
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

        self.train_prepared_path = self.prepared_dir_path + '/sentiment.train_prepared_transformer.pkl'
        self.test_prepared_path = self.prepared_dir_path + '/sentiment.test_prepared_transformer.pkl'
        self.validate_prepared_path = self.prepared_dir_path + '/sentiment.validate_prepared_transformer.pkl'

        self.encoder: Optional[LabelEncoder] = None
        self.encoder_path = self.prepared_dir_path + '/sentiment.encoder.pkl'
        self.encoder_checksum_path = self.prepared_dir_path + '/sentiment.encoder.pkl.md5'

        self.tokenizer: Optional[PreTrainedTokenizer] = AutoTokenizer.from_pretrained(self.transformer_model)

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
        # Prepare encoder
        #
        train_cleaned_checksum = checksum_of_file(self.train_cleaned_path)
        train_cleaned_checksum_exists = read_file_to_string(self.train_cleaned_checksum_path)
        test_cleaned_checksum = checksum_of_file(self.test_cleaned_path)
        test_cleaned_checksum_exists = read_file_to_string(self.test_cleaned_checksum_path)
        validate_cleaned_checksum = checksum_of_file(self.validate_cleaned_path)
        validate_cleaned_checksum_exists = read_file_to_string(self.validate_cleaned_checksum_path)

        if (train_cleaned_checksum != train_cleaned_checksum_exists) or (os.path.isfile(self.encoder_path) is False):
            df = pd.read_csv(self.train_cleaned_path)

            self.encoder = LabelEncoder()
            self.encoder.fit(df['sentiment'].to_numpy())

            write_object_to_file(self.encoder_path, self.encoder)
        else:
            self.encoder = read_file_to_object(self.encoder_path)

        #
        # Prepare data
        #
        if train_cleaned_checksum != train_cleaned_checksum_exists or (os.path.isfile(self.train_prepared_path) is False):
            self.__prepare_dataset(self.train_cleaned_path, self.train_prepared_path, tokenizer)

        if test_cleaned_checksum != test_cleaned_checksum_exists or (os.path.isfile(self.test_prepared_path) is False):
            self.__prepare_dataset(self.test_cleaned_path, self.test_prepared_path, tokenizer)

        if validate_cleaned_checksum != validate_cleaned_checksum_exists or (os.path.isfile(self.validate_prepared_path) is False):
            self.__prepare_dataset(self.validate_cleaned_path, self.validate_prepared_path, tokenizer)

        #
        # Update checksum of cleaned data files
        #
        write_string_to_file(self.train_cleaned_checksum_path, checksum_of_file(self.train_cleaned_path))
        write_string_to_file(self.test_cleaned_checksum_path, checksum_of_file(self.test_cleaned_path))
        write_string_to_file(self.validate_cleaned_checksum_path, checksum_of_file(self.validate_cleaned_path))

    def __clean_dataset(self, input_file: str, input_checksum_file: str, output_file: str) -> None:
        input_checksum = checksum_of_file(input_file)
        input_checksum_exists = read_file_to_string(input_checksum_file)

        if (input_checksum != input_checksum_exists) or (os.path.isfile(output_file) is False):
            df = pd.read_csv(input_file)
            df['review'] = df['review'].progress_apply(lambda s: preprocess_text(s, self.transform_lang, self.transform_pipeline, [clean_html]))
            df.to_csv(output_file)

    def __prepare_dataset(self, input_file: str, output_file: str, tokenizer: PreTrainedTokenizer) -> None:
        print(f"Preparing ${output_file} ...")
        df = pd.read_csv(input_file)
        x = tokenizer(df['review'].tolist(), return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        y = self.encoder.transform(df['sentiment'])
        dataset = SentimentDataset(x, y)
        write_object_to_file(output_file, dataset)

    def setup(self, stage: Optional[str] = None) -> None:
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
        return [
            self.transfer_x_to_device(batch[0], device),
            batch[1].to(device)]

    def transfer_x_to_device(self, batch: BatchEncoding, device: Optional[torch.device] = None) -> BatchEncoding:
        return {
            'input_ids': batch['input_ids'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }

    def preprocess(self, x: List[str], device: torch.device) -> BatchEncoding:
        cleaned = [preprocess_text(s, self.transform_lang, self.transform_pipeline, [clean_html]) for s in x]
        x = self.tokenizer(cleaned, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        return self.transfer_x_to_device(x, device)
