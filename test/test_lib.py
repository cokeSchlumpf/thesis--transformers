import spacy
from lib.utils import extract_sentence_tokens
from lib.text_similarity import most_similar_sentences, similarity

SAMPLE = "james best, best known for his portrayal of bumbling sheriff rosco p. coltrane on tv's the dukes of hazzard, died monday after a brief illness. he was 88. best died in hospice in hickory, north carolina, of complications from pneumonia, said steve latshaw, a longtime friend and hollywood colleague. although he'd been a busy actor for decades in theater and in hollywood, best didn't become famous until 1979, when the dukes of hazzard's cornpone charms began beaming into millions of american homes almost every friday night. for seven seasons, best's rosco p. coltrane chased the moonshine-running duke boys back and forth across the back roads of fictitious hazzard county, georgia, although his hot pursuit usually ended with him crashing his patrol car. although rosco was slow-witted and corrupt, best gave him a childlike enthusiasm that got laughs and made him endearing. his character became known for his distinctive kew-kew-kew chuckle and for goofy catchphrases such as cuff 'em and stuff 'em! upon making an arrest. among the most popular shows on tv in the early 80s, the dukes of hazzard ran until 1985 and spawned tv movies, an animated series and video games. several of best's hazzard co-stars paid tribute to the late actor on social media. i laughed and learned more from jimmie in one hour than from anyone else in a whole year, co-star john schneider, who played bo duke, said on twitter. give uncle jesse my love when you see him dear friend. jimmy best was the most constantly creative person i have ever known, said ben jones, who played mechanic cooter on the show, in a facebook post. every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions. born jewel guy on july 26, 1926, in powderly, kentucky, best was orphaned at 3 and adopted by armen and essa best, who renamed him james and raised him in rural indiana. best served in the army during world war ii before launching his acting career."


class TestUtils:

    def test_sentence_tokenizer(self):
        lang = spacy.load('en_core_web_sm')
        result = extract_sentence_tokens(lang, SAMPLE)

        print(result)


class TestSimilarity:

    def test_sequence_similarity(self):
        lang = spacy.load('en_core_web_sm')

        samples = [
            'My dog is running fast',
            'My car drives fast',
            'My cat is walking slow',
            'I just wrote a letter to Santa Clause'
        ]

        print([similarity(samples[0], samples[i], lang) for i in range(0, len(samples))])

    def test_most_similar_sentences(self):
        s1 = '''James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent 'Return of the Killer Shrews,' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we've lost in 2015 . CNN's Stella Chan contributed to this story.'''
        s2 = '''James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 . "Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.'''
        lang = spacy.load('en_core_web_sm')

        print(most_similar_sentences(s1, s2, lang))
