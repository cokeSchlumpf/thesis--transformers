import spacy
from lib.oracle_summary import extract_oracle_summary
from lib.text_similarity import most_similar_sentences, similarity

from rouge_score import rouge_scorer


class TestOracleSummary:
    
    def test_summary(self):
        print()
        lang = spacy.load('en_core_web_sm')
        source = '''James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent 'Return of the Killer Shrews,' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we've lost in 2015 . CNN's Stella Chan contributed to this story.'''
        target = '''James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88. "Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.'''
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'])

        result = extract_oracle_summary(source, target, lang, oracle_length=True)
        print(result)
        print(similarity(target, result[0], lang))
        print(scorer.score(target, result[0]))


        print('-' * 50)
        result = most_similar_sentences(source, target, lang, similarity_threshold=0.4)
        print(result)

        print(similarity(target, result[0], lang))
        print(scorer.score(target, result[0]))