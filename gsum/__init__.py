import multiprocessing as mp
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


SOURCE_SAMPLES = [
    '''Singer-songwriter David Crosby hit a jogger with his car Sunday evening, a spokesman said. The accident happened in Santa Ynez, California, near where Crosby lives. Crosby was driving at approximately 50 mph when he struck the jogger, according to California Highway Patrol Spokesman Don Clotworthy. The posted speed limit was 55. The jogger suffered multiple fractures, and was airlifted to a hospital in Santa Barbara, Clotworthy said. His injuries are not believed to be life threatening. "Mr. Crosby was cooperative with authorities and he was not impaired or intoxicated in any way. Mr. Crosby did not see the jogger because of the sun," said Clotworthy. According to the spokesman, the jogger and Crosby were on the same side of the road. Pedestrians are supposed to be on the left side of the road walking toward traffic, Clotworthy said. Joggers are considered pedestrians. Crosby is known for weaving multilayered harmonies over sweet melodies. He belongs to the celebrated rock group Crosby, Stills & Nash. "David Crosby is obviously very upset that he accidentally hit anyone. And, based off of initial reports, he is relieved that the injuries to the gentleman were not life threatening," said Michael Jensen, a Crosby spokesman. "He wishes the jogger a very speedy recovery."''',
    '''(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze after Britain were disqualified for a faulty handover. The 26-year-old Bolt has now collected eight gold medals at world championships, equaling the record held by American trio Carl Lewis, Michael Johnson and Allyson Felix, not to mention the small matter of six Olympic titles. The relay triumph followed individual successes in the 100 and 200 meters in the Russian capital. "I'm proud of myself and I'll continue to work to dominate for as long as possible," Bolt said, having previously expressed his intention to carry on until the 2016 Rio Olympics. Victory was never seriously in doubt once he got the baton safely in hand from Ashmeade, while Gatlin and the United States third leg runner Rakieem Salaam had problems. Gatlin strayed out of his lane as he struggled to get full control of their baton and was never able to get on terms with Bolt. Earlier, Jamaica's women underlined their dominance in the sprint events by winning the 4x100m relay gold, anchored by Shelly-Ann Fraser-Pryce, who like Bolt was completing a triple. Their quartet recorded a championship record of 41.29 seconds, well clear of France, who crossed the line in second place in 42.73 seconds. Defending champions, the United States, were initially back in the bronze medal position after losing time on the second handover between Alexandria Anderson and English Gardner, but promoted to silver when France were subsequently disqualified for an illegal handover. The British quartet, who were initially fourth, were promoted to the bronze which eluded their men's team. Fraser-Pryce, like Bolt aged 26, became the first woman to achieve three golds in the 100-200 and the relay. In other final action on the last day of the championships, France's Teddy Tamgho became the third man to leap over 18m in the triple jump, exceeding the mark by four centimeters to take gold. Germany's Christina Obergfoll finally took gold at global level in the women's javelin after five previous silvers, while Kenya's Asbel Kiprop easily won a tactical men's 1500m final. Kiprop's compatriot Eunice Jepkoech Sum was a surprise winner of the women's 800m. Bolt's final dash for golden glory brought the eight-day championship to a rousing finale, but while the hosts topped the medal table from the United States there was criticism of the poor attendances in the Luzhniki Stadium. There was further concern when their pole vault gold medalist Yelena Isinbayeva made controversial remarks in support of Russia's new laws, which make "the propagandizing of non-traditional sexual relations among minors" a criminal offense. She later attempted to clarify her comments, but there were renewed calls by gay rights groups for a boycott of the 2014 Winter Games in Sochi, the next major sports event in Russia.''',
    '''James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career.''',
]


def run_extractive():
    mp.set_start_method('spawn')
    
    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg, is_extractive=True)

    mdl = GuidedExtSum(cfg, dat)

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        max_epochs=5,
        default_root_dir='./data/checkpoints')

    trainer.fit(mdl, dat)
    trainer.test(mdl, dat.test_dataloader())


def run_abstractive():
    mp.set_start_method('spawn')
    date_time = datetime.now().strftime("%Y-%m-%d-%H%M")

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum(cfg, dat)

    training_path = f'./data/trained/{date_time}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=training_path,
        filename='gsum-abs-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=0 if cfg.is_debug else 1,
        default_root_dir=training_path,
        val_check_interval=0.25,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_callback])

    trainer.fit(mdl, dat)
    print('Best model ' + checkpoint_callback.best_model_path)
    trainer.test(mdl, dat.test_dataloader())


def run_test():
    checkpoint_path = './data/trained/2021-07-06-2305/gsum-abs-epoch=04-val_loss=107.72.ckpt'

    cfg = GuidedSummarizationConfig()
    dat = GuidedSummarizationDataModule(cfg)
    mdl = GuidedAbsSum.load_from_checkpoint(checkpoint_path, cfg=cfg, data_module=dat)
    mdl.freeze()

    print(f'Loaded checkpoint from path {checkpoint_path} ...')
    outputs = mdl(SOURCE_SAMPLES)

    for i in range(len(outputs)):
        print('SOURCE')
        print(SOURCE_SAMPLES[i])
        print()
        print('SUMMARY')
        print(outputs[i])
        print()
        print('-' * 50)
        print()
