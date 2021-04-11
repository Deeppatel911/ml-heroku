# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:09:58 2021

@author: Deepnil Vasava
"""

import numpy as np
import pandas as pd
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

emails = pd.read_csv('preprocessed1.csv')
em = emails.dropna(axis=0)
em.sample(3)