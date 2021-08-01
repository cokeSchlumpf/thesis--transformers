from gsum.scoring import run_scoring_abs

if __name__ == "__main__":
    # BERT, MLSum, Oracle
    #run_scoring_abs('./data/trained/2021-07-30-0034/gsum-abs-epoch=04-val_loss=1181.25.ckpt')

    # mBERT, MLSum, Oracle
    #run_scoring_abs('./data/trained/2021-07-30-1237/gsum-abs-epoch=03-val_loss=782.19.ckpt')

    # BERT, Swisstext, Oracle
    #run_scoring_abs('./data/trained/2021-07-30-0330/gsum-abs-epoch=06-val_loss=1681.08.ckpt')

    # mBERT, Swisstext, Oracle
    #run_scoring_abs('./data/trained/2021-07-30-1123/gsum-abs-epoch=07-val_loss=1194.46.ckpt')


    # BERT, spon_ard, Oracle
    #run_scoring_abs('./data/trained/2021-07-31-1923/gsum-abs-epoch=29-val_loss=4424.72.ckpt')

    # mBERT, spon_ard, Oracle
    #run_scoring_abs('./data/trained/2021-07-31-1309/gsum-abs-epoch=33-val_loss=3578.28.ckpt')

    # electra, MLSum, Oracle
    run_scoring_abs('./data/trained/2021-07-31-0635/gsum-abs-epoch=08-val_loss=803.48.ckpt')
