from lib.oracle_summary import extract_oracle_summary
from lib.text_preprocessing import clean_html, preprocess_text, simple_punctuation_only, to_lower
from lib.text_similarity import most_similar_sentences
from lib.textrank import get_keywords, get_summary
from lib.utils import extract_sentence_tokens
from transformers import PreTrainedTokenizer
from typing import List

import re
import spacy
import torch


PAD = '[PAD]'  # padding token
INPUT_SEP = '[SEP]'  # segment separation token
INPUT_CLS = '[CLS]'  # sentence classification token
TARGET_BOS = '[unused0]'  # begin of sequence token
TARGET_EOS = '[unused1]'  # end of sequence token
TARGET_SEP = '[unused2]'  # separator token


class GuidedSummarizationInput:

    token_ids: torch.Tensor
    segment_ids: torch.Tensor
    attention_mask: torch.Tensor
    cls_indices: torch.Tensor
    cls_mask: torch.Tensor

    def __init__(self,
                 token_ids: torch.Tensor,
                 segment_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 cls_indices: torch.Tensor,
                 cls_mask: torch.Tensor):

        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.attention_mask = attention_mask
        self.cls_indices = cls_indices
        self.cls_mask = cls_mask

    def to_dict(self):
        return {
            'token_ids': self.token_ids,
            'segment_ids': self.segment_ids,
            'attention_mask': self.attention_mask,
            'cls_indices': self.cls_indices,
            'cls_mask': self.cls_mask
        }


class GuidedSummarizationTarget:

    token_ids: torch.Tensor

    attention_mask: torch.Tensor

    def __init__(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.token_ids = token_ids
        self.attention_mask = attention_mask

    def to_dict(self):
        return {
            'token_ids': self.token_ids,
            'attention_mask': self.attention_mask
        }


class GuidedSummarizationExtractiveTarget:

    sentence_ids: torch.Tensor

    sentence_mask: torch.Tensor

    sentence_padding_mask: torch.Tensor

    summary: str

    def __init__(self, sentence_ids: torch.Tensor, sentence_mask: torch.Tensor, sentence_padding_mask: torch.Tensor, summary: str):
        self.sentence_ids = sentence_ids
        self.sentence_mask = sentence_mask
        self.sentence_padding_mask = sentence_padding_mask
        self.summary = summary

    def to_dict(self):
        return {
            'sentence_ids': self.sentence_ids,
            'sentence_mask': self.sentence_mask,
            'sentence_padding_mask': self.sentence_padding_mask,
            'summary': self.summary
        }


def preprocess_extractive_output_sample(source: str, sample: str, lang: spacy.Language, max_length: int = 512, max_input_sentences: int = 128, min_sentence_tokens: int = 5, method: str = 'similarity') -> GuidedSummarizationExtractiveTarget:
    """
    Preprocesses targets for extractive summarization.

    :param source The text which should be summarized
    :param sample The summarized source
    :param lang A spaCy language model which is used for text-processing
    :param max_length The maximum input length which is expected by the input encoder
    :param max_input_sentences The max. number of sentences from the input to be used
    :param min_sentence_tokens The minimum length of sentences to be included in the encoded input.
    :param method Which method should be used for extracting sentences? Possible options are 'similarity' or 'oracle'.
    """

    def remove_trailing_punctuation(sents: List[List[str]]):
        """
        After splitting sample into sentences. Each sentence still contains its final punctuation.
        This function removes it.
        """
        res = []
        for sent in sents:
            if sent[-1] == '.' or sent[-1] == '!' or sent[-1] == '?':
                res += [sent[0:-1]]
            else:
                res += [sent]

        return res

    def add_control_tokens(sents: List[List[str]]):
        """
        Each sentence is wrapped with a [CLS] token upfront and a trailing [SEP] token to distinguish between
        sentences.
        """
        res = []
        for sent in sents:
            res += [[INPUT_CLS] + sent + [INPUT_SEP]]

        return res

    def flatten_and_shorten(sents: List[List[str]]) -> str:
        """
        This function flattens the list of sentences to a single list of tokens.
        """
        # sample_tokens will contain all tokens of the sample
        output = ""
        token_count = 0

        for i, sent in enumerate(sents):
            if (token_count + len(sent) <= max_length) and (len(sent) >= min_sentence_tokens):
                token_count += len(sent)
                output += ' '.join(sent)
            elif len(sent) < min_sentence_tokens:
                # Just skip the sentence
                pass
            else:
                # Stop preparing sample.
                break

        output = output.replace(INPUT_CLS, '')
        output = output.replace(f' {INPUT_SEP}', '.')
        output = output.replace(f' ,', ',')
        output = output.replace(f' \'', '\'')
        output = re.sub(r'\s([:?.!,"](?:\s|$))', r'\1', output)
        return output

    pipeline = [simple_punctuation_only, to_lower]
    source_cleaned = preprocess_text(source, lang, pipeline, [clean_html])
    sample_cleaned = preprocess_text(sample, lang, pipeline, [clean_html])

    source_sentences = extract_sentence_tokens(lang, source_cleaned)
    source_sentences = remove_trailing_punctuation(source_sentences)
    source_sentences = add_control_tokens(source_sentences)
    source_cleaned = flatten_and_shorten(source_sentences)

    if method == 'similarity':
        summary, sentence_ids, sentence_mask = most_similar_sentences(source_cleaned, sample_cleaned, lang, max_input_sentences)
    else:
        assert method == 'oracle'
        summary, sentence_ids, sentence_mask = extract_oracle_summary(source_cleaned, sample_cleaned, lang, max_input_sent_count=max_input_sentences, oracle_length=True)

    sentence_padding_mask = [0] * len(sentence_ids)
    sentence_padding_mask += [1] * (max_input_sentences - len(sentence_ids))
    sentence_padding_mask = sentence_padding_mask[:max_input_sentences]

    sentence_ids += [-1] * (max_input_sentences - len(sentence_ids))
    sentence_ids = sentence_ids[:max_input_sentences]

    return GuidedSummarizationExtractiveTarget(torch.Tensor(sentence_ids).type(torch.IntTensor),
                                               torch.Tensor(sentence_mask).type(torch.IntTensor),
                                               torch.Tensor(sentence_padding_mask).type(torch.IntTensor), summary)


def preprocess_output_sample(sample: str, lang: spacy.Language, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> GuidedSummarizationTarget:
    """
    Preprocessing targets for abstractive summarization.

    :param sample The reference summary
    :param lang A spaCy language model which is used for pre-processing
    :param tokenizer The pre-trained BERT tokenizer
    :param max_length Length of output sequence after processing (shorter inputs will be padded, longer sequences will be cut), only full sentences are includes
    """

    def add_control_tokens(sents: List[List[str]]):
        """
        Each sentence is wrapped with a [CLS] token upfront and a trailing [SEP] token to distinguish between
        sentences.
        """
        res = [TARGET_BOS]

        for i, sent in enumerate(sents):
            if len(res) + len(sent) >= (max_length - 1):
                break

            if i > 0:
                res += [TARGET_SEP] + sent
            else:
                res += sent

        tkns = res + [TARGET_EOS]
        msk = ([1] * len(tkns)) + ([0] * (max_length - len(tkns)))
        tkns = tkns + ([PAD] * (max_length - len(tkns)))
        return tkns, msk

    pipeline = [simple_punctuation_only, to_lower]
    cleaned = preprocess_text(sample, lang, pipeline, [clean_html])
    sentences = extract_sentence_tokens(lang, cleaned)
    tokens, attention_mask = add_control_tokens(sentences)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids + ([0] * (max_length - len(tokens)))

    return GuidedSummarizationTarget(
        torch.Tensor(token_ids).type(torch.LongTensor),
        torch.tensor(attention_mask).type(torch.IntTensor))


def preprocess_input_sample(
        sample: str, lang: spacy.language, tokenizer: PreTrainedTokenizer,
        max_length: int = 512, max_input_sentences: int = 128, min_sentence_tokens: int = 5) -> GuidedSummarizationInput:
    """
    Preprocesses a sample for a BERT summarization task. According to [^Liu2019] and [^Dou2021]

    :param sample The input sample as string
    :param lang A spaCy language model which is used for pre-processing
    :param tokenizer The pre-trained BERT tokenizer
    :param max_length Length of input sequence after processing (shorter inputs will be padded, longer sequences will be cut), only full sentences are includes
    :param max_input_sentences Maximum number of input sentences.
    :param min_sentence_tokens Minimum number of tokens a sentence should contain to be included

    Returns:
        A triplet of
        - token_ids: tokens from the input plus control tokens e.g. [CLS], [SEP]
        - segment_ids: additional encoded segment tokens, toggling between 0 and 1 for each sentence
        - cls_indices: the indices which contain a [CLS] token as these indices will contain information for the whole segment
    """

    def remove_trailing_punctuation(sents: List[List[str]]):
        """
        After splitting sample into sentences. Each sentence still contains its final punctuation.
        This function removes it.
        """
        res = []
        for sent in sents:
            if sent[-1] == '.' or sent[-1] == '!' or sent[-1] == '?':
                res += [sent[0:-1]]
            else:
                res += [sent]

        return res

    def add_control_tokens(sents: List[List[str]]):
        """
        Each sentence is wrapped with a [CLS] token upfront and a trailing [SEP] token to distinguish between
        sentences.
        """
        res = []
        for sent in sents:
            res += [[INPUT_CLS] + sent + [INPUT_SEP]]

        return res

    def flatten_and_segments(sents: List[List[str]]):
        """
        This function flattens the list of sentences to a single list of tokens. Additionally a segement
        embedding is created to encode sentence boundaries.
        """
        # sample_tokens will contain all tokens of the sample
        sample_tokens = []
        # sample_segments will contain a toggling (0 or 1) segment embedding
        sample_segments = []

        for i, sent in enumerate(sents):
            if (len(sample_tokens) + len(sent) <= max_length) and (len(sent) >= min_sentence_tokens):
                sample_tokens += sent
                sample_segments += [i % 2] * len(sent)
            elif len(sent) < min_sentence_tokens:
                # Just skip the sentence
                pass
            else:
                # Stop preparing sample.
                break

        return sample_tokens, sample_segments

    def append_right_padding(tkn_ids, sgmt_ids):
        """
        Appends padding to fill arrays up to a length of `max_length`.
        Returns token_ids, segment_ids, attention_mask
        """
        padding = [0] * (max_length - len(tkn_ids))
        att_mask = ([1] * len(tkn_ids)) + padding
        tkn_ids += padding
        sgmt_ids += padding

        return tkn_ids, sgmt_ids, att_mask

    def create_cls(tkns):
        """
        Creates a list of indices of sentence classifier positions ('[CLS]')
        within the sample. Additionally a mask is created to hide unnecessary (non-classifier) indices.

        Returns:
            cls_indices (1 x max_length), cls_mask (1 x max_length)
        """

        indices = [i for i, token in enumerate(tkns) if token == INPUT_CLS]
        indices = indices[:max_input_sentences]

        mask = ([0] * len(indices)) + ([1] * (max_input_sentences - len(indices)))
        indices = indices + ([-1] * (max_input_sentences - len(indices)))

        return indices, mask

    pipeline = [simple_punctuation_only, to_lower]
    cleaned = preprocess_text(sample, lang, pipeline, [clean_html])
    sentences = extract_sentence_tokens(lang, cleaned)
    sentences = remove_trailing_punctuation(sentences)
    sentences = add_control_tokens(sentences)

    tokens, segment_ids = flatten_and_segments(sentences)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids, segment_ids, attention_mask = append_right_padding(token_ids, segment_ids)
    cls_indices, cls_mask = create_cls(tokens)

    return GuidedSummarizationInput(
        torch.Tensor(token_ids).type(torch.IntTensor),
        torch.Tensor(segment_ids).type(torch.IntTensor),
        torch.Tensor(attention_mask).type(torch.IntTensor),
        torch.Tensor(cls_indices).type(torch.IntTensor),
        torch.Tensor(cls_mask).type(torch.IntTensor))


def preprocess_guidance_keywords(source: str, lang: spacy.Language, tokenizer: PreTrainedTokenizer, max_length: int = 512,
                                 max_input_sentences: int = 128, min_sentence_tokens: int = 5) -> GuidedSummarizationInput:
    """
    Prepares a guidance signal by generating keywords from the source.

    :param source The source text.
    :param lang The expected language of the text.
    :param tokenizer The tokenizer used for tokenizing the input signal.
    :param max_length The maximum length expected for the guidance signal.
    :param max_input_sentences The max. number of expected input sentences.
    :param min_sentence_tokens The minimum number of tokens in a sentence to be included.

    Returns:
        The encoded guidance signal.
    """

    keywords = get_keywords(source, lang.lang)
    keywords = ' '.join(keywords)
    return preprocess_input_sample(keywords, lang, tokenizer, max_length, max_input_sentences, min_sentence_tokens)


def preprocess_guidance_extractive(source: str, lang: spacy.Language, tokenizer: PreTrainedTokenizer, max_length: int = 512,
                                   max_input_sentences: int = 128, min_sentence_tokens: int = 5) -> GuidedSummarizationInput:
    """
    Prepares a guidance signal by creating an un-supervised extractive summary using textrank.

    :param source The source text.
    :param lang The expected language of the text.
    :param tokenizer The tokenizer used for tokenizing the input signal.
    :param max_length The maximum length expected for the guidance signal.
    :param max_input_sentences The max. number of expected input sentences.
    :param min_sentence_tokens The minimum number of tokens in a sentence to be included.

    Returns:
        The encoded guidance signal.
    """

    summary = get_summary(source, language=lang.lang)
    return preprocess_input_sample(summary, lang, tokenizer, max_length, max_input_sentences, min_sentence_tokens)


def preprocess_guidance_extractive_training(source: str, target: str, lang: spacy.language, tokenizer: PreTrainedTokenizer,
                                            max_length: int = 512, max_input_sentences: int = 128, min_sentence_tokens: int = 5, method: str = 'oracle') -> GuidedSummarizationInput:
    """
    Prepares an extractive guidance signal for training. By generating an extractive summary based on the target summary.

    :param source The source text.
    :param target The reference summary.
    :param lang SpaCy language model for text processing.
    :param tokenizer The tokenizer used for tokenizing the input signal.
    :param max_length The maximum length expected for the guidance signal.
    :param max_input_sentences The max. number of expected input sentences.
    :param min_sentence_tokens The minimum number of tokens in a sentence to be included.
    :param method The method used to create a summary.

    Returns:
        Encoded Guidance signal.
    """

    if method == 'similarity':
        summary, _, _ = most_similar_sentences(source, target, lang, max_input_sentences)
    else:
        assert method == 'oracle'
        summary, _, _ = extract_oracle_summary(source, target, lang, max_input_sent_count=max_input_sentences, oracle_length=True)

    return preprocess_input_sample(summary, lang, tokenizer, max_length, max_input_sentences, min_sentence_tokens)
