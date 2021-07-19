import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import spacy
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import GuidedSummarizationConfig
from .preprocess_inputs import preprocess_extractive_output_sample, preprocess_guidance_extractive_training, preprocess_input_sample, preprocess_output_sample, GuidedSummarizationInput, GuidedSummarizationExtractiveTarget, GuidedSummarizationTarget
from lib.utils import checksum_of_file, read_file_to_object, read_file_to_string, write_object_to_file, write_string_to_file

tqdm.pandas()

#SPACY_MODEL = spacy.load('en_core_web_md')


class GuidedSummarizationDataset(Dataset):

    def __init__(self,
                 x_input: List[GuidedSummarizationInput],
                 x_guidance: any,
                 y: Optional[Union[List[GuidedSummarizationTarget], List[GuidedSummarizationExtractiveTarget]]] = None):

        self.x_input = x_input
        self.x_guidance = x_guidance
        self.y = y

    def __len__(self):
        return len(self.x_input)

    def __getitem__(self, idx):
        if self.y is not None:
            result = {
                'x_input': self.x_input[idx].to_dict(),
                'y': self.y[idx].to_dict()
            }
        else:
            result = {
                'x_input': self.x_input[idx].to_dict()
            }

        return result


class SummaryDataset(Dataset):

    def __init__(self, source: List[str], references: Optional[List[str]] = None):
        if references is not None:
            assert len(references) == len(source)

        self.source = source
        self.references = references

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index) -> Tuple[str, Optional[str]]:
        if self.references is not None:
            return self.source[index], self.references[index]
        else:
            return self.source[index], None


class GuidedSummarizationDataModule(pl.LightningDataModule):

    def __init__(self, config: GuidedSummarizationConfig, is_extractive: bool = False):
        super(GuidedSummarizationDataModule, self).__init__()
        self.config = config
        self.lang = spacy.load(config.spacy_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        self.is_extractive = is_extractive

        self.train: Optional[GuidedSummarizationDataset] = None
        self.test: Optional[GuidedSummarizationDataset] = None
        self.validate: Optional[GuidedSummarizationDataset] = None

    def prepare_data(self) -> None:
        #
        # Preprocess raw data including tokenization and cleansing.
        #
        self.__prepare_dataset('test')
        self.__prepare_dataset('train')
        self.__prepare_dataset('validation')
        print("Preparation done.")

    def preprocess(self, x: List[str]) -> DataLoader:
        #
        # Pre-processes input sequences for prediction
        #
        x_prepared = [preprocess_input_sample(sample, self.lang, self.tokenizer, self.config.max_input_length, self.config.max_input_sentences, self.config.min_sentence_tokens) for sample in x]
        dataset = GuidedSummarizationDataset(x_prepared, x_prepared)
        return DataLoader(dataset, num_workers=0)

    def preprocess_target(self, x: List[str]) -> DataLoader:
        y = [preprocess_output_sample(sample, self.lang, self.tokenizer, self.config.max_target_length) for sample in x]
        x_prepared = [preprocess_input_sample(sample, self.lang, self.tokenizer, self.config.max_input_length, self.config.max_input_sentences, self.config.min_sentence_tokens) for sample in x]
        dataset = GuidedSummarizationDataset(x_prepared, x_prepared, y)
        return DataLoader(dataset, num_workers=0)

    def prepare_source(self, params: (str, int, pd.DataFrame)) -> str:
        """
        Processes a batch, stores the result in a file and returns the file path.
        """
        dataset, batch_idx, df = params
        path = self.config.data_prepared_path + '/' + dataset + '.prepared.' + str(batch_idx) + '.source.pkl'
        result = df['text']\
            .progress_apply(lambda s: preprocess_input_sample(s, self.lang, self.tokenizer, self.config.max_input_length, self.config.max_input_sentences, self.config.min_sentence_tokens))\
            .to_list()

        write_object_to_file(path, result)
        return path

    def prepare_guidance_extractive(self, params: (str, int, pd.DataFrame)) -> str:
        """
        Process a batch, store the result in a file and returns the path.
        """
        dataset, batch_idx, df = params

        path = self.config.data_prepared_path + '/' + dataset + '.prepared.' + str(batch_idx) + '.guidance.ext.' + self.config.extractive_preparation_method + '.pkl'
        result = [
            preprocess_guidance_extractive_training(row['text'], row['summary'], self.lang, self.tokenizer, self.config.max_input_sentences, self.config.max_input_sentences, method=self.config.extractive_preparation_method)
            for idx, row in tqdm(df.iterrows(), total=len(df.index))]
        write_object_to_file(path, result)
        return path

    def prepare_extractive_target(self, params: (str, int, pd.DataFrame)) -> str:
        """
        Processes a batch, stores the result in a file and returns the file path.
        """
        dataset, batch_idx, df = params
        path = self.config.data_prepared_path + '/' + dataset + '.prepared.' + str(batch_idx) + '.target.ext.' + self.config.extractive_preparation_method + '.pkl'
        if os.path.isfile(path):
            return path
        else:
            result = [preprocess_extractive_output_sample(row['text'], row['summary'], self.lang, self.config.max_input_sentences, self.config.extractive_preparation_method) for idx, row in tqdm(df.iterrows(), total=len(df.index))]
            write_object_to_file(path, result)
            return path

    def prepare_target(self, params: (str, int, pd.DataFrame)) -> str:
        """
        Processes a batch, stores the result in a file and returns the file path.
        """
        dataset, batch_idx, df = params
        path = self.config.data_prepared_path + '/' + dataset + '.prepared.' + str(batch_idx) + '.target.pkl'
        result = df['summary']\
            .progress_apply(lambda s: preprocess_output_sample(s, self.lang, self.tokenizer, self.config.max_target_length))\
            .to_list()
        write_object_to_file(path, result)
        return path

    def __prepare_dataset(self, dataset: str = 'test') -> None:
        print(f'Preparing dataset {dataset} ...')
        # Make sure that the target directory exists
        Path(self.config.data_prepared_path).mkdir(parents=True, exist_ok=True)

        #
        # Load raw data
        #
        raw_data_checksum_path = self.config.data_prepared_path + '/' + dataset + '.raw.md5'
        raw_data_checksum_latest = read_file_to_string(raw_data_checksum_path)

        raw_source_path = self.config.data_raw_path + '/' + dataset + '.source'
        raw_target_path = self.config.data_raw_path + '/' + dataset + '.target'

        raw_source_checksum = checksum_of_file(raw_source_path)
        raw_target_checksum = checksum_of_file(raw_target_path)

        raw_source_df = pd.read_csv(raw_source_path)
        raw_target_df = pd.read_csv(raw_target_path)

        raw_data = pd.concat([raw_source_df, raw_target_df], axis=1)  # [:200]  # TODO
        raw_data_checksum = raw_source_checksum + raw_target_checksum
        write_string_to_file(raw_data_checksum_path, raw_data_checksum)

        batches = enumerate(np.array_split(raw_data, mp.cpu_count()))
        batches = list(map(lambda t: (dataset, t[0], t[1]), batches))

        #
        # Prepare source data
        #
        prepared_source_path = self.config.data_prepared_path + '/' + dataset + '.prepared.source.pkl'
        if (os.path.isfile(prepared_source_path) is False) or (raw_data_checksum != raw_data_checksum_latest):
            print('> process source data')
            with mp.Pool(mp.cpu_count()) as p:
                results = p.map(self.prepare_source, batches)

            prepared_target = []
            for result in results:
                prepared_target += list(read_file_to_object(result))
                os.remove(result)

            print(f'> processed {len(prepared_target)} samples')
            write_object_to_file(prepared_source_path, prepared_target)

        #
        # Prepare target data
        #
        if not self.is_extractive:
            prepared_target_path = self.config.data_prepared_path + '/' + dataset + '.prepared.target.pkl'
            if (os.path.isfile(prepared_target_path) is False) or (raw_data_checksum != raw_data_checksum_latest):
                print('> process target data')
                with mp.Pool(mp.cpu_count()) as p:
                    results = p.map(self.prepare_target, batches)

                prepared_target = []
                for result in results:
                    prepared_target += read_file_to_object(result)
                    os.remove(result)

                print(f'> processed {len(prepared_target)} samples')
                write_object_to_file(prepared_target_path, prepared_target)

        #
        # Prepare extractive target data
        #
        if self.is_extractive:
            prepared_ext_target_path = self.config.data_prepared_path + '/' + dataset + '.prepared.target.ext.' + self.config.extractive_preparation_method + '.pkl'
            if (os.path.isfile(prepared_ext_target_path) is False) or (raw_data_checksum != raw_data_checksum_latest):
                print('> process extractive target data')

                with mp.Pool(mp.cpu_count()) as p:
                    results = p.map(self.prepare_extractive_target, batches)

                prepared_ext_target = []
                for result in results:
                    prepared_ext_target += read_file_to_object(result)
                    os.remove(result)

                print(f'> processed {len(prepared_ext_target)} samples')
                write_object_to_file(prepared_ext_target_path, prepared_ext_target)

        #
        # Prepare extractive summary as guidance signal
        #
        if self.config.guidance_method == 'extractive':
            prepared_guidance_ext_path = self.config.data_prepared_path + '/' + dataset + '.prepared.guidance.ext.' + self.config.extractive_preparation_method + '.pkl'

            if (os.path.isfile(prepared_guidance_ext_path) is False) or (raw_data_checksum != raw_data_checksum_latest):
                print('< process guidance signals for extractive summary')

                with mp.Pool(8) as p:
                    results = p.map(self.prepare_guidance_extractive, batches)

                prepared_guidance_ext = []
                for result in results:
                    prepared_guidance_ext += read_file_to_object(result)
                    os.remove(result)

                print(f'> processed {len(prepared_guidance_ext)} samples')
                write_object_to_file(prepared_guidance_ext_path, prepared_guidance_ext)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = self.__setup_dataloader('train')
        self.test = self.__setup_dataloader('test')
        self.validate = self.__setup_dataloader('validation')
        print(f'Setup data module with summarization dataset.')

    def __setup_dataloader(self, dataset: str = 'test'):
        print(f'Loading dataset `{dataset}` ...')
        prepared_source_path = self.config.data_prepared_path + '/' + dataset + '.prepared.source.pkl'
        prepared_target_path = self.config.data_prepared_path + '/' + dataset + '.prepared.target.pkl'
        prepared_ext_target_path = self.config.data_prepared_path + '/' + dataset + '.prepared.target.ext.' + self.config.extractive_preparation_method + '.pkl'

        prepared_source: List[GuidedSummarizationInput] = read_file_to_object(prepared_source_path)

        if self.is_extractive:
            prepared_target: List[GuidedSummarizationExtractiveTarget] = read_file_to_object(prepared_ext_target_path)
        else:
            prepared_target: List[GuidedSummarizationTarget] = read_file_to_object(prepared_target_path)

        #
        # Remove empty samples as they would lead to errors during training.
        #
        remove_indices = []

        for i in range(len(prepared_source)):
            if torch.sum(prepared_source[i].token_ids) == 0:
                remove_indices.append(i)

            if self.is_extractive and torch.sum(prepared_target[i].sentence_ids) == 0:
                remove_indices.append(i)
            elif not self.is_extractive and torch.sum(prepared_target[i].to_dict()['token_ids']) == 0:
                remove_indices.append(i)

        if len(remove_indices) > 0:
            print(f'Removing indices due to invalid samples {remove_indices} ...')

        for i in reversed(remove_indices):
            del prepared_target[i]
            del prepared_source[i]

        return GuidedSummarizationDataset(prepared_source, 0, prepared_target)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if self.train is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            return DataLoader(self.train, self.config.batch_sizes[0], num_workers=0, shuffle=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.test is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            return DataLoader(self.test, self.config.batch_sizes[1], num_workers=0)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.validate is None:
            raise RuntimeError('DataModule not setup properly. Call `setup` before accessing datasets.')
        else:
            return DataLoader(self.validate, self.config.batch_sizes[2], num_workers=0)

    def inference_dataloader(self, dataset: str = 'test', shuffle: bool = True) -> DataLoader:
        """
        Returns data for inference. The data is not pre-processed, the data loader also contains the reference sample for
        calculating ROUGE scores later.

        :param dataset The name of the dataset to load the data from.
        :param shuffle Whether data should be shuffled while loading.
        """
        raw_source_path = self.config.data_raw_path + '/' + dataset + '.source'
        raw_target_path = self.config.data_raw_path + '/' + dataset + '.target'

        raw_source_df = pd.read_csv(raw_source_path)
        raw_target_df = pd.read_csv(raw_target_path)

        raw_data = pd.concat([raw_source_df, raw_target_df], axis=1)
        sources = [row['text'] for _, row in raw_data.iterrows()]
        references = [row['summary'] for _, row in raw_data.iterrows()]

        return DataLoader(SummaryDataset(sources, references), batch_size=self.config.batch_sizes[3], num_workers=0, shuffle=shuffle)

    def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        batch['x_input']['token_ids'] = batch['x_input']['token_ids'].to(device)
        batch['x_input']['segment_ids'] = batch['x_input']['segment_ids'].to(device)
        batch['x_input']['attention_mask'] = batch['x_input']['attention_mask'].to(device)
        batch['x_input']['cls_indices'] = batch['x_input']['cls_indices'].to(device)
        batch['x_input']['cls_mask'] = batch['x_input']['cls_mask'].to(device)

        if self.is_extractive:
            batch['y']['sentence_ids'] = batch['y']['sentence_ids'].to(device)
            batch['y']['sentence_mask'] = batch['y']['sentence_mask'].to(device)
            batch['y']['sentence_padding_mask'] = batch['y']['sentence_padding_mask'].to(device)
        else:
            batch['y']['token_ids'] = batch['y']['token_ids'].to(device)
            batch['y']['attention_mask'] = batch['y']['attention_mask'].to(device)

        return batch
