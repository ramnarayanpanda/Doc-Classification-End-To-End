from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



class FeatureExtractor:

    def __init__(self, ngram_range=(1, 1)):
        self.label_encoder = None
        self.count_encoding = None
        self.tf_idf_encoding = None
        self.ngram_range = ngram_range


    def transform(self, data):
        
        self.label_encode(data)
        self.count_features(data)
        self.tfidf_vectorizer(data[['text']], ngram_range=self.ngram_range)


    def label_encode(self, data):
        """Label encodes different document categories"""
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(data[['class']])


    def count_features(self, data, is_plot=False):
        '''Takes dataframe as input. Then concatenates all the doc in the text field of the DF.
        Counts all the occurrences of each word in the all docs. Var = count_words
        Sorts the Counter list in decreasing order. Var = sorted_words

        Based on the counts of occurrences the highest count will get assigned to 1, lowest will be length of unique words.

        Then for each word in each doc return the integer mapping from the above step
        '''
        all_text = '\n'.join(list(data['text']))
        tokenizer = RegexpTokenizer(r'\w+')

        all_words = [word for word in tokenizer.tokenize(all_text)]
        count_words = Counter(all_words)
        total_words = len(all_words)
        print('total words length', total_words)
        sorted_words = count_words.most_common(total_words)
        print('sorted words length', len(sorted_words))

        self.count_encoding = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
        print('Count encoding dict length',len(self.count_encoding))


    def tfidf_vectorizer(self, data, data_type='train', ngram_range=(1, 1)):
        """This preprocesses the data with tfidf vectorizer"""
        self.tf_idf_encoding = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
        self.tf_idf_encoding.fit(data['text'])  # this generates a sparse matrix