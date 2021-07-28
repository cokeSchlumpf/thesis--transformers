from pydantic import BaseModel, parse_file_as

from typing import Optional, Tuple


class GuidedSummarizationConfig(BaseModel):
    """
    Indicates whether executing in debug mode, if true, GPU will be disabled an data loaders are not working
    parallelized to avoid issues with PyCharm debugger.
    """
    is_debug: bool = False

    """
    Data configuration
    """
    data_raw_path: str = './data/raw/mlsum'

    data_prepared_path: str = './data/prepared/mlsum/bert-base-german-dbmdz-uncased'

    """
    Pre-preprocessing configuration
    """
    spacy_model: str = 'de_core_news_md'  # Alternatives: 'en_core_web_sm', 'en_core_web_trf'

    extractive_preparation_method: str = 'similarity'  # Alternatives: 'similarity', 'oracle'

    """
    Training batch sizes (training, test, validation, inference)
    """
    batch_sizes: Tuple[int, int, int, int] = (20, 20, 20, 50)

    """
    Number of batches to accumulate during training.
    """
    accumulate_grad_batches: int = 1

    """
    The name of the huggingface transformer base model.
    """
    base_model_name: str = 'bert-base-german-dbmdz-uncased'  # 'bert-base-uncased'

    """
    Reuse a pretrained Bert module from another (usually extractive) taining
    """
    base_model_pretrained: Optional[str] = None

    """
    The maximum length (token count) for input sequences.
    """
    max_input_length: int = 512

    """
    The maximum length (token count) for target sequences.
    """
    max_target_length: int = 256

    """
    The maximum length for signal sequences.
    """
    max_input_signal_length: int = 512

    """
    The maximum number of sentences to be considered for document embeddings.
    """
    max_input_sentences: int = 32

    """
    The minimum length for sentences to be considered.
    """
    min_sentence_tokens: int = 4

    """
    Parameters for EncoderTransformerLayer 
    """
    encoder_heads: int = 8

    encoder_ff_dim: int = 2048

    encoder_dropout: float = 0.1  # 0.2

    """
    Parameters for DecoderTransformerLayer
    """
    decoder_layers: int = 6

    decoder_heads: int = 8

    decoder_dim: int = 768

    decoder_ff_dim: int = 2048

    decoder_dropout: float = 0.2

    """
    Label Smoothing used within loss function during training
    """
    label_smoothing: float = 0.1

    """
    Encoder Optimizer Settings
    """
    encoder_optim_lr: float = 2e-3

    encoder_optim_beta: Tuple[float, float] = (0.9, 0.999)

    encoder_optim_warmup_steps: int = 10000  # 20000

    encoder_optim_eps: float = 1e-9

    """
    Decoder Optimizer Settings
    """
    decoder_optim_lr: float = 0.1

    decoder_optim_beta: Tuple[float, float] = (0.9, 0.999)

    decoder_optim_warmup_steps: int = 8000  # 10000

    decoder_optim_eps: float = 1e-9

    """
    Beam Search configuration
    """
    beam_k: int = 3

    beam_alpha: float = 0.75

    """
    Guidance configuration
    """
    guidance_method: Optional[str] = 'extractive'  # `extractive`, `keywords`

    """
    Maximum number of parallel processes during preparation.
    """
    max_cpus: int = 64

    """
    The number of batches used during pre-training.
    """
    max_batches: int = 32

    @staticmethod
    def from_file(file: str) -> 'GuidedSummarizationConfig':
        return parse_file_as(GuidedSummarizationConfig, file)

    @staticmethod
    def apply(
            dataset: str,
            base_model: str,
            extractive: bool,
            guidance_signal: Optional[str] = None,
            extractive_preparation_method: str = 'oracle',
            debug: bool = False,
            base_model_pretrained: Optional[str] = None) -> 'GuidedSummarizationConfig':
        """
        Helper method to create configuration class.

        :param dataset The name of the dataset `cnn_dailymail`, `mlsum` or `swisstext`.
        :param base_model The base model to use for the experiment (`bert` or `distilbert`)
        :param extractive Whether the configuration is for extractive model or not.
        :param guidance_signal The guidance signal used for the experiment. None, `extractive` or `keywords`.
        :param extractive_preparation_method The method to be used to prepare extractive summarization target.
        :param debug Whether it is a debug run or not.
        :param base_model_pretrained An optional pre-trained base model (BERT part) which should be used. This would replace the other configure base_model. Only valid for abstractive training (base model then is usually pre-trained with extractive model)
        """

        if dataset == 'cnn_dailymail':
            lang = 'en'
        elif dataset == 'mlsum':
            lang = 'de'
        elif dataset == 'swisstext':
            lang = 'de'
        else:
            raise Exception('unknown dataset')

        if base_model == 'bert' and lang == 'en':
            mdl = 'bert-base-uncased'
        elif base_model == 'bert' and lang == 'de':
            mdl = 'bert-base-german-dbmdz-uncased'
        elif base_model == 'distilbert' and lang == 'en':
            mdl = 'distilbert-base-cased'
        elif base_model == 'distilbert' and lang == 'de':
            mdl = 'distilbert-base-german-cased'
        elif base_model == 'electra' and lang == 'en':
            mdl = 'google/electra-base-discriminator'
        elif base_model == 'electra' and lang == 'de':
            mdl = 'german-nlp-group/electra-base-german-uncased'
        elif base_model == 'multilingual':
            mdl = 'bert-base-multilingual-uncased'
        else:
            raise Exception('Unknown model/language combination.')

        cfg = GuidedSummarizationConfig()
        cfg.is_debug = debug
        cfg.base_model_name = mdl
        cfg.data_raw_path = f'./data/raw/{dataset}'
        cfg.data_prepared_path = f'./data/prepared/{dataset}{"_debug" if debug else ""}/{mdl}'
        cfg.spacy_model = 'en_core_web_sm' if lang == 'en' else 'de_core_news_md'

        if extractive:
            cfg.accumulate_grad_batches = 4
            cfg.encoder_optim_warmup_steps = 10000
            cfg.decoder_optim_warmup_steps = 8000
            cfg.batch_sizes = (20, 20, 20, 50)
        else:
            cfg.accumulate_grad_batches = 5
            cfg.encoder_optim_warmup_steps = 20000
            cfg.decoder_optim_warmup_steps = 10000
            cfg.batch_sizes = (8, 8, 8, 25)
            cfg.base_model_pretrained = base_model_pretrained

        if base_model == 'distilbert':
            cfg.encoder_optim_lr = 2e-3
            cfg.decoder_optim_lr = 0.05

        cfg.guidance_method = guidance_signal
        cfg.extractive_preparation_method = extractive_preparation_method

        return cfg
