def train_extractive():
    mp.set_start_method('spawn')

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    mdl = GuidedExtSum(cfg, dat)

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        max_epochs=5,
        default_root_dir='./data/checkpoints')

    trainer.fit(mdl, dat)
    trainer.test(mdl, dat.test_dataloader())