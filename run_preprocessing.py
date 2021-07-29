from gsum.config import GuidedSummarizationConfig
from gsum.summarizer import GuidedSummarizationDataModule

if __name__ == "__main__":
    config = GuidedSummarizationConfig.apply('mlsum', 'multi', True, guidance_signal='oracle')
    dat = GuidedSummarizationDataModule(config, is_extractive=False)

    dat.prepare_data()
