import yake

from summa.summarizer import summarize
from typing import List


def get_summary(s: str, ratio: float = 0.2, language: str = 'en') -> str:
    if language == 'en':
        language = 'english'
    elif language == 'de':
        language = 'german'

    selected = summarize(s, ratio=ratio, language=language, split=True)
    return ' '.join(selected)


def get_keywords(s: str, language: str = 'en') -> List[str]:
    keywords = yake.KeywordExtractor(lan=language, n=2, top=5).extract_keywords(s)
    keywords = sorted(keywords, key=lambda t: t[1], reverse=True)
    keywords = list(map(lambda t: t[0], keywords))
    return keywords
