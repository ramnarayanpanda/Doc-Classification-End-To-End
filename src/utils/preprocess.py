import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, wordpunct_tokenize
from nltk.stem import *
from bs4 import BeautifulSoup
import unidecode
import pandas as pd
import numpy as np
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer



class Preprocessor:

    def __init__(self, remove_special=True, remove_stop=True, stem=False, 
                 lemm=True, summy=False, stem_type='Snow Ball', lemm_type='Word Net'):
        self.remove_special = remove_special
        self.remove_stop = remove_stop
        self.summy = summy
        
        self.stem = stem
        self.lemm = lemm
        self.stem_type = stem_type
        self.lemm_type = lemm_type


    def transform(self, text):
        if self.stem==True and self.lemm==True:
            raise ValueError(f"Choose either lemming or stemming. Choosing both is not ideal.")
        
        if self.remove_special==True:
             text = self.remove_specialCharacters(text, self.summy)

        if self.remove_stop==True:
            text = self.remove_stopWords(text)

        if self.stem==True:
            text = self.stemming(text, stem_type=self.stem_type)
            
        if self.lemm==True:
            text = self.lemming(text, lemm_type=self.lemm_type)

        return text


    def remove_specialCharacters(self, text, summy):
        """This function removes all special characters.
        First replaces %$#,'().- with space, then iterate over each word, if the word contains anything other than A-Za-z remove those
        For now we have removed numbers as well. This may need more analysis."""
        
        # Convert to lower case first
        text = str(text).lower()

        # Remove html tags if any, adding the below line actually increasing the unique words from 18613 to 18628,
        # so commented out the below line, may need to re visit in case the data changes
        text = BeautifulSoup(text, 'html.parser').get_text()

        # Remove accented characters
        # adding the below also does not change anything, may the corpus does not contain any accented characters
        text = unidecode.unidecode(text)

        # Try to add contractions map as well, currently facing issue with pip install contractions
        eng_vocab = set(nltk.corpus.words.words())
        text = ' '.join([word for word in re.sub(r'\s+', ' ', re.sub(r"[!%$#,'().-]", ' ', text)).split(' ')
                         if len(word) > 2 and word.isalpha()])
        
        if summy:
            print('\n>>>>>>>>>>>>>>>text summarization applied\n')
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 5)
            text = ' '.join([str(i) for i in summary])

        return text


    def remove_stopWords(self, text):
        """Removes stop words, First text is converted to lower case, then tokenized, stopword is applied"""
        
        # I, the, am, are, not, are,
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        text = ' '.join([word for word in tokenizer.tokenize(text) if word not in stop_words])
        return text


    def stemming(self, text, stem_type='Snow Ball'):
        # Boys -> Boy, Collections -> Collect, Stemming -> Stem, Doing -> Do
        
        available_stemmers = ['Snow Ball', 'Porter', 'Lancaster']
        if stem_type not in available_stemmers:
            raise ValueError(f"Only {available_stemmers} stemming are available.")

        if stem_type == 'Snow Ball':
            stemmer = SnowballStemmer(language='english')
        elif stem_type == 'Porter':
            stemmer = PorterStemmer()
        elif stem_type == 'Lancaster':
            stemmer = LancasterStemmer()

        tokenizer = RegexpTokenizer(r'\w+')
        text = ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(text)])

        return text
    
    
    def lemming(self, text, lemm_type='Word Net'):
        
        available_lemmers = ['Word Net']
        if lemm_type not in available_lemmers:
            raise ValueError(f"Only {available_lemmers} lemming are available.")
            
        
        lemmatizer = WordNetLemmatizer()
        
        tokenizer = RegexpTokenizer(r'\w+')
        text = ' '.join([lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text)])
        
        return text 
        