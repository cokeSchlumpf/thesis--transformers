import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .data import SentimentDataModule
from .model import SentimentClassifier


def run():
    sentiment_data = SentimentDataModule()
    sentiment_classifier = SentimentClassifier(sentiment_data)

    """
    sentiment_data.prepare_data()
    sentiment_data.setup()
    
    batch = next(iter(sentiment_data.test_dataloader()))
    print(batch)
    result = sentiment_classifier.training_step(batch, 0)
    print(result)
    """

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=30,
        default_root_dir='./data/checkpoints',
        callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=5)])

    trainer.fit(sentiment_classifier, sentiment_data)
    trainer.test(sentiment_classifier, sentiment_data.test_dataloader())

    model_path = './models/sentiment_transformers_classifier.mdl'
    sentiment_classifier.freeze()
    torch.save(sentiment_classifier, model_path)

    sentiment_classifier = torch.load(model_path)
    sentiment_classifier.eval()

    print(sentiment_classifier([
        'Basically there\'s a family where a little boy (Jake) thinks there\'s a zombie in his closet & his parents are fighting all the time.',
        'I sure would like to see a resurrection of a up dated Seahunt series with the tech they have today it would bring back the kid excitement in me.',
        'This movie is boring and stupid, I am sure I have never seen much more bullshit in two hours of my life.']))
