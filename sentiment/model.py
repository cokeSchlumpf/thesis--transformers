import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from typing import List

from .data import SentimentDataModule


class SentimentClassifier(pl.LightningModule):

    def __init__(self, data_module: SentimentDataModule):
        super().__init__()

        self.dim = data_module.vector_dimensions
        self.data_module = data_module

        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2))

    def forward(self, x: List[str]):
        x_prepared = self.data_module.preprocess(x, self.device)
        pred = self.predict(x_prepared)
        pred_val_encoded = torch.argmax(pred, dim=1)
        return self.data_module.encoder.inverse_transform(pred_val_encoded.numpy())

    def predict(self, x: torch.Tensor):
        return self.model(F.pad(x, [0, self.dim - x.shape[1]], mode='constant', value=0))

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
        return torch.optim.Adam(self.parameters(), lr=0.002)