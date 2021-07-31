"""
Code from partly taken from https://github.com/nlpyang/BertSum/blob/master/LICENSE and
https://github.com/pltrdy/extoracle_summarization/blob/master/LICENSE
"""
import numpy as np
import re
import rouge
import spacy

from lib.utils import extract_sentence_tokens
from rouge.rouge_score import Ngrams
from typing import List, Tuple


def extract_oracle_summary(src: str, tgt: str, lang: spacy.Language, min_sentence_length: int = 5, summary_length: int = 5, max_input_sent_count: int = 5, oracle_length: bool = False) -> Tuple[str, List[int], List[int]]:
    def _clean_line(line):
        return line.strip()

    if not isinstance(src, str):
        src = ''

    if not isinstance(tgt, str):
        tgt = ''

    src = _clean_line(src)
    tgt = _clean_line(tgt)

    src_sentences = extract_sentence_tokens(lang, src)
    src_sentences = list(filter(lambda s: len(s) >= min_sentence_length, src_sentences))
    tgt_sentences = extract_sentence_tokens(lang, tgt)

    src_sentences = src_sentences[:max_input_sent_count]

    if oracle_length:
        summary_length = len(tgt_sentences)

    ids, sents = greedy_selection(src_sentences, tgt_sentences, summary_length)
    summary = '. '.join([' '.join(sents[i]) for i in ids])
    classification_result = np.zeros(max_input_sent_count)
    classification_result[ids] = 1

    return summary, ids, classification_result


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def _get_ngrams(n: int, text: str, exclusive: bool = True) -> Ngrams:
    """
    Calculates n-grams.

    :param n Which n-grams to calculate
    :param text A list of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = Ngrams(exclusive=exclusive)
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def cal_rouge(evaluated_ngrams: Ngrams, reference_ngrams: Ngrams) -> dict:
    """
    Calculate Rouge score between two sets of n_grams.

    :param evaluated_ngrams first set
    :param reference_ngrams second set

    Returns:
        Rouge scores between the two sets.
    """
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    return rouge.rouge_score.f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)


def greedy_selection(doc_sent_list: List[List[str]], abstract_sent_list: List[List[str]],
                     summary_size: int, exclusive_ngrams: bool = False) -> Tuple[List[int], List[List[str]]]:
    """Greedy ext-oracle on lists of sentences

    :param doc_sent_list The text document tokenized as a list of sentences.
    :param abstract_sent_list The reference summary tokenized as a list of sentences.
    :param summary_size The maximum length of the expected summary.
    :param exclusive_ngrams Whether ngrams should be distinct/ exclusive. Default is False.

    Returns:
        The oracle summary.
    """
    def _get_word_ngrams(n, sentences):
        return _get_ngrams(n, sentences, exclusive=exclusive_ngrams)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, sent) for sent in sents]
    reference_1grams = _get_word_ngrams(1, abstract)
    evaluated_2grams = [_get_word_ngrams(2, sent) for sent in sents]
    reference_2grams = _get_word_ngrams(2, abstract)

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue

            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = Ngrams.union(*candidates_1)
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = Ngrams.union(*candidates_2)
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if cur_id == -1:
            return selected, sents

        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), sents
