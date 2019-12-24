import re
import csv

import random
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import opinion_lexicon


NEGATIVE_WORDS = set(opinion_lexicon.negative())
HTTP_PATTERN = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')
MENTION_PATTERN = re.compile('(@|#)\w+')
CHARS_PATTERN = re.compile('[^a-zA-z\s]|^b\s+')


def preprocessing(df):
    def handle_links(text):
        text = re.sub(HTTP_PATTERN, ' ', text)
        return text

    def handle_mentions(text):
        text = re.sub(MENTION_PATTERN, ' ', text)
        return text

    def handle_special_chars(text):
        text = re.sub(CHARS_PATTERN, '', text)
        return text

    def process(text):
        text = text.lower()
        text = handle_links(text)
        text = handle_mentions(text)
        text = handle_special_chars(text)
        return text

    df['tweet_'] = df['tweet'].apply(process)
    return df


def classify(df):
    vectorizer = TfidfVectorizer(binary=False, max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'))
    tfidf_model = vectorizer.fit(df['tweet_'].values)
    X = tfidf_model.transform(df['tweet_'].values).toarray()
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    predictions = text_classifier.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    df = pd.read_csv("data.csv", sep="\t")
    df = preprocessing(df)
    classify(df)
