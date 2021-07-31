from gsum.config import GuidedSummarizationConfig
from gsum.summarizer import GuidedSummarizationDataModule

if __name__ == "__main__":
    config = GuidedSummarizationConfig.apply('mlsum', 'bert', False, guidance_signal='extractive')
    dat = GuidedSummarizationDataModule(config)

    dat.prepare_data()
