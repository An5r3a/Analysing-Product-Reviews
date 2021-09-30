import Feature_extraction as fe
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
#from readability import Readability
import math
import enchant
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import shap
from torch.nn.utils.weight_norm import weight_norm
import torch 
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import model_zoo
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import distance, pairwise_distances


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

warnings.filterwarnings('ignore')

fun_clean = lambda x: pd.Series(fe.text_cleaning(x))
fun_pos = lambda x: pd.Series(fe.part_of_speech(x))
fun_flesch = lambda x: pd.Series(fe.flesch_reading(x))
fun_dale_score = lambda x: pd.Series(fe.dale_chall(x))
fun_difficult_words = lambda x: pd.Series(fe.difficult_words(x))
fun_text_length = lambda x: pd.Series(fe.text_length(x))
fun_set_length = lambda x: pd.Series(fe.set_length(x))
fun_wrong_words = lambda x: pd.Series(fe.wrong_words(x))
fun_num_letters = lambda x: pd.Series(fe.frequency_counter(x))
fun_entropy = lambda x: pd.Series(fe.calculate_entropy(x))
fun_rating = lambda x: pd.Series(fe.rating(x))
fun_help = lambda x,y: pd.Series(fe.map_helpfulness_value(x,idx))

# Extract Features
df_cleantext = df.loc[:]['customer_reviews'].apply(fun_clean)
df['customer_reviews_clean'] = df_cleantext

# Features 1 - 3
df_pos = df.loc[:]['customer_reviews_clean'].apply(fun_pos)
df_pos.columns = ['Noun','Adjective','Verb']

df['Noun'] = df_pos['Noun']
df['Adjective'] = df_pos['Adjective']
df['Verb'] = df_pos['Verb']

# Feature 4
df_flesch = df.loc[:]['customer_reviews_clean'].apply(fun_flesch)
df_flesch.columns = ['flesch_score', 'flesch_ease']
df['flesch_score'] = df_flesch['flesch_score']
df['flesch_ease'] = df_flesch['flesch_ease']

# Feature 5
df_dale = df.loc[:]['customer_reviews_clean'].apply(fun_dale_score)
df_dale.columns = ['dale_score', 'dale_ease']
df['dale_score'] = df_dale['dale_score']
df['dale_grade_levels'] = df_dale['dale_ease']

# Feature 6
df_diff = df.loc[:]['customer_reviews_clean'].apply(fun_difficult_words)
df['difficult_words'] = df_diff

# Feature 7
df_length = df.loc[:]['customer_reviews_clean'].apply(fun_text_length)
df['text_length'] = df_length

# Feature 8
df_set = df.loc[:]['customer_reviews_clean'].apply(fun_set_length)
df['text_set_length'] = df_set

# Feature 9
df_wrong = df.loc[:]['customer_reviews_clean'].apply(fun_wrong_words)
df['wrong_words'] = df_wrong

# Feature 10-12
df_letters = df.loc[:]['customer_reviews_clean'].apply(fun_num_letters)
df_letters.columns = ['one_letter','two_letters','more_letters']
df['one_letter'] = df_letters['one_letter']
df['two_letters'] = df_letters['two_letters']
df['more_letters'] = df_letters['more_letters']

# Feature 13
df['Lex_diversity'] = df['text_set_length'] / df ['text_length']

# Feature 14
df_entropy = df.loc[:]['customer_reviews_clean'].apply(fun_entropy)
df['entropy'] = df_entropy

# Feature 15
df_rating = df.loc[:]['average_review_rating'].apply(rating)
df['rating'] = df_rating

# Calculate similarity
df['desc_clean'] = df.loc[:]['description'].apply(fun_clean)
df['prodinfo_clean'] = df.loc[:]['product_information'].apply(fun_clean)
df['proddesc_clean'] = df.loc[:]['product_description'].apply(fun_clean)

descKW = np.unique(df['desc_clean'])
prodinfoKW = np.unique(df['prodinfo_clean'])
proddescKW = np.unique(df['proddesc_clean'])
sbKW = np.concatenate((descKW,prodinfoKW, proddescKW), axis = None)

count = CountVectorizer()
bag_of_words_st = count.fit_transform(sbKW)
bw_st = bag_of_words_st.toarray()

dist_st = 1- distance.cdist(bw_st, bw_st , metric = 'cosine')

df_sim = df
words_len = df.shape[0]
similarity_descinfo = []
similarity_desc = []
similarity_prod_desc_info = []

for i in range(words_len):
    kw1 = list(sbKW).index(df_sim.loc[i]['desc_clean'])
    kw2 = list(sbKW).index(df_sim.loc[i]['prodinfo_clean'])
    sim = dist_st[kw1][kw2]
    similarity_descinfo.append((sim))
    

for i in range(words_len):
    kw1 = list(sbKW).index(df_sim.loc[i]['desc_clean'])
    kw2 = list(sbKW).index(df_sim.loc[i]['proddesc_clean'])
    sim = dist_st[kw1][kw2]
    similarity_desc.append((sim))
    
for i in range(words_len):
    kw1 = list(sbKW).index(df_sim.loc[i]['proddesc_clean'])
    kw2 = list(sbKW).index(df_sim.loc[i]['prodinfo_clean'])
    sim = dist_st[kw1][kw2]
    similarity_prod_desc_info.append((sim))

df['sim_descinfo'] = pd.DataFrame(similarity_descinfo)
df['sim_desc'] = pd.DataFrame(similarity_desc)
df['sim_prod_desc_info'] = pd.DataFrame(similarity_prod_desc_info)

# Calculate helpfulness Score

df_features = df[['Noun','Adjective','Verb','text_length','text_set_length','difficult_words',
                  'wrong_words','one_letter', 'two_letters', 'more_letters',
                           'entropy','Lex_diversity', 'sim_descinfo',
       'sim_desc', 'sim_prod_desc_info']]

df_help = df[['Noun','Adjective','Verb','text_length','text_set_length','difficult_words',
                  'wrong_words','one_letter', 'two_letters', 'more_letters',
                           'entropy','Lex_diversity','sim_descinfo',
       'sim_desc', 'sim_prod_desc_info']].describe()

helpfulness_variables = ['Noun','Adjective','Verb','text_length','text_set_length','difficult_words',
                  'wrong_words','one_letter', 'two_letters', 'more_letters',
                           'entropy','Lex_diversity', 'sim_descinfo',
       'sim_desc', 'sim_prod_desc_info']

helpfulness_values = [df_help['Noun']['mean'],df_help['Adjective']['mean'],df_help['Verb']['mean'],df_help['text_length']['mean'],
                      df_help['text_set_length']['mean'],df_help['difficult_words']['mean'],
                  df_help['wrong_words']['mean'],df_help['one_letter']['mean'], df_help['two_letters']['mean'],df_help['more_letters']['mean'],
                      df_help['entropy']['mean'],df_help['Lex_diversity']['mean'],
                      df_help['sim_descinfo']['mean'],df_help['sim_desc']['mean'],
                      df_help['sim_prod_desc_info']['mean']]

