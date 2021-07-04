import multiprocessing as mp
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import GuidedSummarizationConfig
from .data import GuidedSummarizationDataModule
from .summarizer import GuidedAbsSum, GuidedExtSum


SOURCE_SAMPLES = [
    '''James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he'd been a busy actor for decades in theater and in Hollywood, Best didn't become famous until 1979, when "The Dukes of Hazzard's" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best's Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff 'em and stuff 'em!" upon making an arrest. Among the most popular shows on TV in the early '80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best's "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life's many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career.''',
    '''Singer-songwriter David Crosby hit a jogger with his car Sunday evening, a spokesman said. The accident happened in Santa Ynez, California, near where Crosby lives. Crosby was driving at approximately 50 mph when he struck the jogger, according to California Highway Patrol Spokesman Don Clotworthy. The posted speed limit was 55. The jogger suffered multiple fractures, and was airlifted to a hospital in Santa Barbara, Clotworthy said. His injuries are not believed to be life threatening. "Mr. Crosby was cooperative with authorities and he was not impaired or intoxicated in any way. Mr. Crosby did not see the jogger because of the sun," said Clotworthy. According to the spokesman, the jogger and Crosby were on the same side of the road. Pedestrians are supposed to be on the left side of the road walking toward traffic, Clotworthy said. Joggers are considered pedestrians. Crosby is known for weaving multilayered harmonies over sweet melodies. He belongs to the celebrated rock group Crosby, Stills & Nash. "David Crosby is obviously very upset that he accidentally hit anyone. And, based off of initial reports, he is relieved that the injuries to the gentleman were not life threatening," said Michael Jensen, a Crosby spokesman. "He wishes the jogger a very speedy recovery."'''
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
        max_epochs=5,
        default_root_dir=training_path,
        val_check_interval=0.25,
        callbacks=[checkpoint_callback])

    trainer.fit(mdl, dat)
    print('Best model ' + checkpoint_callback.best_model_path)
    trainer.test(mdl, dat.test_dataloader())


def run_test():
    checkpoint_path = './data/trained/2021-07-03-2327/gsum-abs-epoch=00-val_loss=1533.46.ckpt'

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
