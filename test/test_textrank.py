import spacy

from lib.text_preprocessing import preprocess_text, clean_html, to_lower, simple_punctuation_only, lemmatize, remove_stopwords
from lib.textrank import get_keywords, get_summary

SAMPLE = """James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career."""


class TestTextrank:
    
    def test_preprocessing(self):
        lang = spacy.load('en_core_web_sm')
        pipeline = [simple_punctuation_only, to_lower]
        result = preprocess_text(SAMPLE, lang, pipeline)
        summary = get_summary(result)

        print(summary)

    def test_keywords(self):
        lang = spacy.load('en_core_web_sm')
        pipeline = [simple_punctuation_only, to_lower]
        result = preprocess_text(SAMPLE, lang, pipeline)
        keywords = get_keywords(result)

        print(keywords)
