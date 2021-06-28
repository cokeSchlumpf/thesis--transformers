import numpy as np
import spacy

from typing import List


def most_similar_sentences(s1: str, s2: str, lang: spacy.Language, max_input_sent_count: int = 5,
                           similarity_threshold: float = 0.6) -> (str, List[int], np.array):
    """
    This method returns the most similar sentences of `s1` compared to the sentences contained in `s2`.
    Thus the method tries to create an extractive summary of `s1` based on a provided reference summary `s2`.

    Args:
        s1: The full text sample
        s2: A reference summary for s1
        lang: A spaCy language model which is used for processing the text
        max_input_sent_count: Maximum number of sentences to be considered from the input.
        similarity_threshold: A threshold (between 0 and 1), only sentences with a minimum similarity of this threshold will be selected for the extractive summary.

    Returns:
        A triple of:
        - The extractive summary (str),
        - A list of the indices of the selected sentences (List[int])
        - A classification mask of size `max_input_sent_count` indicating whether sentence is selected (1) or not (0) (np.array)
    """
    docs1 = [lang(sent.text) for sent in lang(s1).sents]
    docs2 = [lang(sent.text) for sent in lang(s2).sents]

    if len(docs1) > max_input_sent_count:
        docs1 = docs1[:max_input_sent_count]

    similarities = np.zeros([len(docs2), len(docs1)])
    for i in range(0, len(docs2)):
        for j in range(0, len(docs1)):
            similarities[i, j] = docs2[i].similarity(docs1[j])

    selected_idx = []
    for i in range(0, similarities.shape[0]):
        max_idx = np.argmax(similarities[i])
        max_val = np.max(similarities[i])
        selected_idx += [(max_idx, max_val)]
        similarities[:, max_idx] = 0

    selected_idx = filter(lambda t: t[1] > similarity_threshold, selected_idx)
    selected_idx = map(lambda t: t[0], selected_idx)
    selected_idx = list(selected_idx)
    selected_sentences = [docs1[i].text for i in selected_idx]
    selected_sentences = ' '.join(selected_sentences)
    classification_result = np.zeros(max_input_sent_count)
    classification_result[selected_idx] = 1

    return selected_sentences, selected_idx, classification_result


def similarity(s1: str, s2: str, lang: spacy.Language) -> float:
    """
    Returns a similarity score between to sequences.

    Args:
        s1: First sequence
        s2: Second sequence
        lang: A spaCy language model used for text processing.

    Returns:
        A similarity score between 0 and 1.
    """
    doc1 = lang(s1)
    doc2 = lang(s2)

    return doc1.similarity(doc2)
