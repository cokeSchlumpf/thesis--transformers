from gsum.config import GuidedSummarizationConfig
from gsum.summarizer import GuidedSummarizationDataModule

if __name__ == "__main__":
    config = GuidedSummarizationConfig.apply('swisstext', 'bert', False, guidance_signal='oracle')
    dat = GuidedSummarizationDataModule(config)

    dat.prepare_data()
