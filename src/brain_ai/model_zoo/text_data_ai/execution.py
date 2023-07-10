"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from sentiment_analysis import sample_func
"""
# https://huggingface.co/yiyanghkust/finbert-tone

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class SentimentDataExecutor:
    """make it a class and then use it in the main file"""

    def __init__(self):
        self.finbert_tone = self.setup_finbert_tone()
        self.model, self.tokenizer, self.labels = self.setup_twitter_roberta_base_sentiment()

    def setup_finbert_tone(self):
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        finbert_tone = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
        return finbert_tone

    def get_labels_from_finbert_tone(self, text):
        label_score = self.finbert_tone(text)[0]
        # LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

        return {label_score['label']: label_score['score']}

    def setup_twitter_roberta_base_sentiment(self):
        # Tasks:
        # emoji, emotion, hate, irony, offensive, sentiment
        # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

        task = 'sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with open("/home/chandravesh/PhDWork/PycharmProjects/sentiment-analysis/data/mapping.txt", "r") as f:
            html = f.read().split("\n")
            print(html)
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        # TF
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
        model.save_pretrained(MODEL)
        tokenizer.save_pretrained(MODEL)
        return model, tokenizer, labels

    def get_labels_from_twitter_roberta_base_sentiment(self, text):
        roberta_score = dict()

        encoded_input = self.tokenizer(text, return_tensors='tf')
        output = self.model(encoded_input)
        scores = output[0][0].numpy()
        scores = softmax(scores)

        for la, sc in zip(self.labels, scores):
            roberta_score[la] = sc

        return roberta_score

    def execute_all_models(self, text):
        roberta_score = self.get_labels_from_twitter_roberta_base_sentiment(text)
        finbert_score = self.get_labels_from_finbert_tone(text)
        return {"Roberta Score": roberta_score, "FinBert Score": finbert_score}


    def average_sentiment_score(self):
        pass
