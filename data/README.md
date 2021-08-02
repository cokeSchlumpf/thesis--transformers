# Datasets

## Amazon Reviews (amzn)

This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

The raw dataset is available on [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews). Download `Reviews.csv` and place it in `/data/raw/amzn/reviews.csv`. Then run `python ./lib/data/prepare_amzn.py`.

## SwissText 2019 (swisstext)

The dataset was published for the SwissText conference 2019. It was used for a [German text summarization challenge](https://www.swisstext.org/swisstext.org/2019/shared-task/german-text-summarization-challenge.html). 

The draw data can be downloaded [here](https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN).

## CNN/ DailyMail 3.0.0 (cnn_dailymail)

The well-known CNN/ DailyMail data set for text summarization (version 3.0.0). The data has been fetched via [HuggingFace Datasets](https://huggingface.co/datasets/cnn_dailymail).

To prepare the data for this project, run `python ./lib/data/prepare_cnndailymail.py`.