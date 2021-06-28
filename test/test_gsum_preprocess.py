import spacy

from gsum.preprocess import preprocess_input_sample, preprocess_output_sample
from transformers import AutoTokenizer, PreTrainedTokenizer

SOURCE_SAMPLE = """James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career."""


class TestGSumPreprocess:

    def test_preprocess_input_sample(self):
        lang = spacy.load('en_core_web_sm')
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        max_length = 512
        result = preprocess_input_sample(SOURCE_SAMPLE, lang, tokenizer, max_length)
        print(tokenizer.convert_ids_to_tokens(result.token_ids.tolist()))
        print(result.segment_ids)
        print(result.cls_indices)
        print(result.attention_mask)
        print(tokenizer.convert_ids_to_tokens([result.token_ids[i] for i in result.cls_indices]))

        assert(result.segment_ids.shape[0] == max_length)
        assert(result.attention_mask.shape[0] == max_length)
        assert(result.token_ids.shape[0] == max_length)

        max_length = 32
        result = preprocess_input_sample(SOURCE_SAMPLE, lang, tokenizer, max_length)
        assert (result.segment_ids.shape[0] == max_length)
        assert (result.attention_mask.shape[0] == max_length)
        assert (result.token_ids.shape[0] == max_length)

    def test_preprocess_output_sample(self):
        lang = spacy.load('en_core_web_sm')
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        output = preprocess_output_sample(SOURCE_SAMPLE, lang, tokenizer)
        print(tokenizer.convert_ids_to_tokens(output.token_ids.tolist()))
