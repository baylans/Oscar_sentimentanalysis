import pandas as pd
import numpy as np
import datetime
import tweepy
import re
import spacy
import glob
import gensim
import gensim.corpora as corpora
import spacy
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import string
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from textblob import TextBlob
import datetime
from datetime import datetime

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from cleantext import clean

import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain

class WildfireFunctions:
    def __init__(self, location):
        self.location = location
        wildfire = pd.read_csv(self.location)
        wildfire.reset_index(drop=True, inplace=True)
        wildfire=wildfire.dropna() 
        self.tweet= wildfire["tweets"]
        self.date= wildfire["date"]
        self.rt= wildfire["rt"]
        self.fav= wildfire["fav"]

    def load_file(self):
        wildfire = pd.read_csv(self.location)
        wildfire.reset_index(drop=True, inplace=True)
        self.wildfire=wildfire.dropna()    
        return self.wildfire
    
    def remove_links(self, tweet):
        tweet = re.sub(r'https?:\/\/\S+', '', tweet)# remove hyper links
        tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
        tweet = tweet.strip('[links]') # remove links
        return tweet

    def remove_text(self, tweet):
        tweet = re.sub('@[A-Za-z0-9-_]+', '', tweet) #to remove mentions
        tweet = re.sub(r'#', '', tweet) # to remove hashtags
        tweet = re.sub(r'RT[/s]+', '', tweet) # to remove retweets
        return tweet

    def remove_words(self, tweet):
        mycharacters = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•∑€®₺¥π¨`æ´¬¨∆^∂ßæ≈√∫~µ≤≥÷ƒ@'
        tweet = re.sub('['+mycharacters + ']+', ' ', tweet) # remove characters 
        tweet = re.sub('\s+', ' ', tweet)   #remove double spacing
        tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
        return tweet
    
    def cleaning_tweet(self, tweet, bigrams=False):
        mywords = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
        mystopwords = set(stopwords.words('english'))
        tweet = tweet.lower() # to make lower case
        tweet = self.remove_text(tweet)
        tweet = self.remove_links(tweet)
        tweet = self.remove_words(tweet)
        tweet = clean(tweet, no_emoji=True)
        tweet_token = [word for word in tweet.split(' ')
                        if word not in mystopwords] # remove stopwords

        tweet_token = [mywords(word) if '#' not in word else word
                        for word in tweet_token] # apply word rooter
        if bigrams:
            tweet_token = tweet_token+[tweet_token[i]+'_'+tweet_token[i+1]
                                        for i in range(len(tweet_token)-1)]
        tweet = ' '.join(tweet_token)
        return tweet

    def cleaning_tweets(self):
        self.clean_tweets = self.tweet.apply(self.cleaning_tweet)
        return self.clean_tweets


    def get_top_n_gram(self, corpus,ngram_range,n=None):
        vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]  
    
    def find_mentioned(self, tweet):
        '''This function will extract the twitter handles of people mentioned in the tweet'''
        return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

    def find_hashtags(self, tweet):
        '''This function will extract hashtags'''
        return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)
    
    def get_redundant_pairs(self, df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(self, df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = self.get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        au_corr = au_corr.dropna()
        return au_corr[0:n]
    
    def display_topics(self, model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)
    
    def separate_tweets(self, tweet):
        exlude = set(string.punctuation)
        lemma = WordNetLemmatizer()
        tweet = ''.join(ch for ch in tweet if ch not in exlude)
        tweet = ' '.join([lemma.lemmatize(word) for word in tweet.split()])
        return tweet.split()

    #Creating a function to get the subjectivity
    def getSubjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    # Creating a function to get the polarity
    def getPolarity(self, text):
        return TextBlob(text).sentiment.polarity

    #Create a function to compute the negative, neutral and positive analysis
    def getAnalysis(self, score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'


                

