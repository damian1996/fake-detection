import pandas as pd
import pickle
import re

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer
import xgboost as xgb

class Preprocessing:
    def __init__(self, text):
        self.text = text

    def lower_text(self):
        self.text = str(self.text).lower()

    def lemmatization(self):
        lemmatizer = WordNetLemmatizer()
        self.text = ' '.join([lemmatizer.lemmatize(word) for word in self.text.split()])

    def remove_stopwords(self):
        stoplist = stopwords.words('english')
        self.text = ' '.join([word for word in str(self.text).split() if word not in stoplist])

    def remove_numbers_and_puntuaction(self):
        self.text = re.sub('[^a-zA-Z]', ' ', self.text)

    def full_preprocess(self):
        self.remove_stopwords()
        self.remove_numbers_and_puntuaction()
        self.lemmatization()
        self.lower_text()

        return self.text

def read_text(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f]
    
    return '\n'.join(lines)

def preprocess_test_data(text):
    preprocessed_text = Preprocessing(text).full_preprocess()
    vectorizer = HashingVectorizer()
    text_vectorized = vectorizer.fit_transform([preprocessed_text])
    return text_vectorized

if __name__ == '__main__':
    model_name = 'xgboost_model.pkl'
    with open(model_name, 'rb') as f:
        classifier = pickle.load(f)

    reliable_article = read_text('reliable_text.txt')
    reliable_vectorized = preprocess_test_data(reliable_article)
    print(classifier.predict(reliable_vectorized)[0])

    fake_article = read_text('fake_text.txt')
    fake_vectorized = preprocess_test_data(fake_article)
    print(classifier.predict(fake_vectorized)[0])