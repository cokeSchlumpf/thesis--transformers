from pydantic import BaseModel

from typing import List


class GuidedSummarizationConfig(BaseModel):

    """
    Data configuration
    """
    data_raw_path: str = './data/raw/cnn_dailymail'

    data_prepared_path: str = './data/prepared/cnn_dailymail'

    """
    Pre-preprocessing configuration
    """
    spacy_model: str = 'en_core_web_sm'

    """
    Training batch sizes (training, test, validation)
    """
    batch_sizes: List[int] = [20, 20, 20]

    """
    The name of the huggingface transformer base model.
    """
    base_model_name: str = 'bert-base-uncased'

    """
    The maximum length (token count) for input sequences.
    """
    max_pos: int = 256

    """
    The maximum length (token count) for target sequences.
    """
    max_target_length: int = 256

    """
    The maximum length for signal sequences.
    """
    max_input_signal_length: int = 256

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

    encoder_dropout: float = 0.2

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


