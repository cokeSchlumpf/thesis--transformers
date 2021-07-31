import re
import spacy

from spacy.tokens import Token
from spacy.language import Language
from typing import Callable, List, Optional
from tqdm import tqdm

Token.set_extension('ppt_output', default='')
WS_PATTERN = re.compile(r'\s{2,}')
CLEAN_HTML_PATTERN = re.compile('<.*?>')


def clean_html(s: str) -> str:
    """
    Removes all HTML tags from the string.

    :param s: The input string
    :return: The cleaned string
    """
    if isinstance(s, str):
        return re.sub(CLEAN_HTML_PATTERN, '', s)
    else:
        return ''


def lemmatize(token: Token) -> None:
    """
    Use the lemmatized version of a word.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = f"{token.lemma_}{token.whitespace_}"


def remove_stopwords(token: Token) -> None:
    """
    Deselects stopwords from the text.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    if token.is_stop:
        token._.ppt_output = token.whitespace_


def simple_punctuation_only(token: Token) -> None:
    """
    Deselects tokens which are not alpha-numeric or ',', '.', '?', '!', ':', ';', '-', '(' or ')'

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    cond = len(token.text) == 1 and not (token.is_alpha or token.is_digit or token.text in [',', '.', '?', '!', '(', ')', ':', ';', '-'])

    if cond:
        token._.ppt_output = token.whitespace_


def words_only(token: Token) -> None:
    """
    Deselects every token which is not a word.

    :param token: The spaCy token
    :return Nothing; Token is mutated
    :return:
    """
    if not token.is_alpha:
        token._.ppt_output = token.whitespace_


def to_lower(token: Token) -> None:
    """
    Turns all characters to lower case.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = token._.ppt_output.lower()


def tokenize(token: Token) -> None:
    """
    Will ensure that every token is separated by a whitespace.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = f"{token._.ppt_output} "


def use_text(token: Token) -> None:
    """
    Sets the text of the token as the expected output text. This step must be included in any pipeline.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = token.text_with_ws


def preprocess_text(
        s: str,
        lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None,
        clean: List[Callable[[str], str]] = []) -> str:
    """
    Preprocesses a string with the help of spaCy. Allows different composable pre-processing methods.

    :param s: The string to be pre-processed
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :param clean: An optional clean function which might be executed on text before processing with spaCy
    :return: The pre-processed string
    """
    doc, pipeline = _parse(s, lang, pipeline, clean)

    for task in pipeline:
        for token in doc:
            task(token)

    joined = ''.join([token._.ppt_output for token in doc])
    return re.sub(WS_PATTERN, ' ', joined).strip()


def preprocess_tokens(
        s: str,
        lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None) -> List[str]:

    """
    Preprocesses a string with the help of spaCy. Allows different composable pre-processing methods.

    :param s: The string to be pre-processed
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :return: The pre-processed list of tokens
    """

    doc, pipeline = _parse(s, lang, pipeline)

    for task in pipeline:
        for token in doc:
            task(token)

    return [str.strip(token._.ppt_output) for token in doc if len(str.strip(token._.ppt_output)) > 0]


def preprocess_doc(doc: spacy.language.Doc, pipeline: List[Callable[[Token], None]]) -> str:
    """
    Executes the pre-procesisng pipeline on a spacy doc.

    :param doc: A spacy document
    :param pipeline:
    :return:
    """
    for task in pipeline:
        for token in doc:
            task(token)

    joined = ''.join([token._.ppt_output for token in doc])
    return re.sub(WS_PATTERN, ' ', joined)


def preprocess_all(
        s: List[str], lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None,
        batch_size: Optional[int] = None, n_process: Optional[int] = -1) -> List[str]:
    """
    Preprocesses a list of samples.

    :param s: The list of samples.
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :param batch_size: The batch size while processing the samples
    :param n_process: The number of parallel processes; -1 is CPU cores count
    :return: The pre-processed string
    """

    if lang is None:
        lang = spacy.load('de_dep_news_trf')

    if pipeline is None:
        pipeline = [to_lower, simple_punctuation_only]

    result = [preprocess_doc(doc, pipeline) for doc in tqdm(lang.pipe(s, batch_size=batch_size, n_process=n_process))]

    return result


def _parse(
        s: str, lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None,
        clean: List[Callable[[str], str]] = []):

    if lang is None:
        lang = spacy.load('de_dep_news_trf')

    if pipeline is None:
        pipeline = [to_lower, simple_punctuation_only]

    pipeline = [use_text] + pipeline

    cleaned = s
    for c in clean:
        cleaned = c(cleaned)

    return lang(cleaned), pipeline


if __name__ == '__main__':
    print(preprocess_text('Hello "World" und so!'))
