from summa.summarizer import summarize
from summa import keywords

def get_summary(s: str, ratio: float = 0.3, language: str = 'english'):
    return summarize(s, ratio=ratio, language=language)

def get_keywords(s: str, language: str = 'english'):
    return keywords(s, language=language)

