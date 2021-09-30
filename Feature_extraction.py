import pandas as pd
import numpy as np
import os
import re
import warnings
import yake
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from readability import Readability
import math
import enchant
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
from functools import partial
import itertools
import tqdm
import shutil

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

warnings.filterwarnings('ignore')


def text_cleaning(text):
    
    try:
        # Removing Stopwords
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])

        # Removing Unicode
        # ASCII formats emojis and other non-ASCII characters as Unicode
        # encoding the text to ASCII format
        text_encode = text.encode(encoding="ascii", errors="ignore")
        # decoding the text
        text_decode = text_encode.decode()
        # cleaning the text to remove extra whitespace 
        text = " ".join([word for word in text_decode.split()])

        # Removing URLs, Hashtags, Punctuation, Mentions
        text = re.sub("Product Description", "", text)
        text = re.sub("@\S+", "", text)
        text = re.sub("\$", "", text)
        text = re.sub("https?:\/\/.*[\r\n]*", "", text)
        text = re.sub("#", "", text)
        text = re.sub('\w*\d\w*', ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation),
                      ' ', text.lower())
        text = text.replace('\n', ' ')
        punct = set(string.punctuation) 
        text = "".join([ch for ch in text if ch not in punct])
    except:
        text = ''
      
    return text
    
one_letter = ['a','I']
two_letter = ['of', 'to', 'in','it', 'is', 'be', 'as', 'at', 'so', 'we', 
              'he', 'by', 'or', 'on', 'do', 'if', 'me', 'my', 'up', 'an', 'go', 'no', 'us', 'am']

three_letter = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 
                'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now','old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too','use', 'that', 'with', 'have', 'this', 'will', 'your', 'from', 
                'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time']

def text_length(x):
    return len(x.split())


def part_of_speech(x):
    
    text_pos = TextBlob(x)
    tags = text_pos.tags
    
    noun = 0
    adjective = 0
    verb = 0
    
    for i in range(len(tags)):
        if 'JJ' in tags[i][1]:
            adjective += 1
        elif 'NN' in tags[i][1]: 
            noun += 1
        elif 'VB' in tags[i][1]: 
            verb += 1
    
    return noun, adjective, verb

def frequency_counter(x):
    
    one = 0
    two = 0
    three = 0
    
    words = x.split()
    for w in words:
        if w in one_letter:
            one += 1
        elif w in two_letter: 
            two += 1
        elif w in three_letter:
            three += 1
    
    return one, two, three 

def calculate_entropy(text, normalized: bool = False):
        """
        Calculates entropy and normalized entropy of list of elements that have specific frequency
        :param frequency: The frequency of the elements.
        :param normalized: Calculate normalized entropy
        :return: entropy or (entropy, normalized entropy)
        """
        stemmer_output = stemmer(text)
        lemmatized_output = lemmatizer(stemmer_output)
        
        text_split = lemmatized_output.split()
        frequency = np.unique(text_split, return_counts = True)[1]
        
        entropy, normalized_ent, n = 0, 0, 0
        sum_freq = sum(frequency)
        for i, x in enumerate(frequency):
            p_x = float(frequency[i] / sum_freq)
            if p_x > 0:
                n += 1
                entropy += - p_x * math.log(p_x, 2)
        if normalized:
            if math.log(n) > 0:
                normalized_ent = entropy / math.log(n, 2)
            return entropy, normalized_ent
        else:
            return entropy

def stemmer(x): 
    s_stemmer = SnowballStemmer(language='english')
    words = x.split()
    stemmer_output = ' '.join([ s_stemmer.stem(w) for w in words])
    
    return stemmer_output

def lemmatizer(x):  
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(x)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    return lemmatized_output

def set_length(x):
    stemmer_output = stemmer(x)
    lemmatized_output = lemmatizer(stemmer_output)
    length = lemmatized_output.split()
    unique_words = len(np.unique(length))

    return unique_words

def flesch_reading(x):
    
    try:
        r = Readability(x)
        flesch_score = r.flesch()
    except:
        flesch_score = -1

    return flesch_score

def flesch_reading(x):
    
    try:
        r = Readability(x)
        flesch_score = r.flesch()
        score = flesch_score.score
        ease = flesch_score.ease
    except:
        score = -1
        ease = 'No Score'

    return score, ease


def dale_chall(x):
    
    try:
        r = Readability(x)
        dale_score = r.dale_chall()
        score = dale_score.score
        ease = dale_score.grade_levels[0]
    except:
        score = -1
        ease = 'No Score'

        
    return score, ease

def difficult_words(x):
    
    words = 0
    try:
        with open('difficultWords.txt') as file:
            content = file.read()
            search = x.split()
            for s in search:
                if s in content:
                    words += 1
            diffwords = len(search) - words
    except:
        diffwords = 0
        
    return diffwords
        

def wrong_words(x):
    try:
        d = enchant.Dict("en_US") 
        result = ''.join([i for i in x if not i.isdigit()])
        result = result.split()
        words = 0
        for r in result:
            if d.check(r):
                words += 1

        wrongWords = len(result) - words
    except:
        wrongWords = 0
    
    return wrongWords
    
def rating(x):
    
    try:
        text = x.split()
        star = float(text[0])
        
    except:
        star = 0.0
    return star 

def extract_helpfulness_features(x):
    colnames = x.columns
    num_col = len(colnames)
    binary_df = []
    
    for i in range(num_col):
        col = colnames[i]
        #print(col)
        val_index = helpfulness_variables.index(col)
        val = helpfulness_values[val_index]
        #calculate_help =  np.vectorize(fun_help)(x.loc[:][col],val_index)
        binary_df.append((x.loc[:][col].apply(lambda x: 1 if x > val else 0)))

    return map(list, zip(*binary_df))
