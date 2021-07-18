import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import AutoModel, BatchEncoding, PreTrainedModel
from typing import List

from .data import SentimentDataModule


class ExtractResultFromTransformer(nn.Module):

    def forward(self, x):
        return x['pooler_output']


class SentimentClassifier(pl.LightningModule):

    def __init__(self, data_module: SentimentDataModule):
        super().__init__()

        self.data_module = data_module
        self.transformer: PreTrainedModel = AutoModel.from_pretrained(data_module.transformer_model)


        for name, param in self.transformer.named_parameters():
            param.requires_grad = False

        self.process = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.transformer.config.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, x: List[str]):
        x_prepared = self.data_module.preprocess(x, self.device)
        pred = self.predict(x_prepared)
        pred_val_encoded = torch.argmax(pred, dim=1)
        return self.data_module.encoder.inverse_transform(pred_val_encoded.numpy())

    def predict(self, x: BatchEncoding):
        self.transformer.to(self.device)
        transformer_out = self.transformer(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
        result = self.process(transformer_out['pooler_output'])
        return result

    def shared_step(self, step, batch):
        x, y = batch
        z = self.predict(x)

        loss = F.cross_entropy(z, y)
        z_encoded = torch.argmax(z, dim=1).cpu()
        y_cpu = y.cpu()

        self.log(f'{step}_loss', loss, prog_bar=True)
        self.log(f'{step}_accuracy', accuracy_score(y_cpu, z_encoded), prog_bar=True)
        self.log(f'{step}_f1', f1_score(y_cpu, z_encoded), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step('train', batch)

    def validation_step(self, batch, batch_idx):
        self.shared_step('val', batch)

    def test_step(self, batch, batch_idx):
        self.shared_step('test', batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)
