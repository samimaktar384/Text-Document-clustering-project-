# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:42:48 2017

@author: rohit_000
"""
from __future__ import print_function

import nltk
import re
import glob
import pandas as pd
    
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")



def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
   
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


path=input('Enter the Location:')
file=glob.glob(path)
synopsis=[]

file_list=[i.split('\\')[4] for i in file]

for i in file:
    p=open(i)
    a=p.read()
    synopsis.append(a)

#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopsis:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                            use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))



tfidf_matrix = tfidf_vectorizer.fit_transform(synopsis) #fit the vectorizer to synopses
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
#print(terms)
mat=tfidf_matrix.toarray()
#pd.DataFrame({'words': }, index =file_list})
mat_pd=pd.DataFrame(data=mat,index=file_list,columns=terms)
mat_pd.to_csv('document',sep='\t')
#print(mat_pd)"""
#print('done')

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
from sklearn.cluster import KMeans

num_clusters = 2

km = KMeans(n_clusters=num_clusters) 
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
from sklearn.externals import joblib
joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

doc = { 'title': file_list,  'synopsis': synopsis, 'cluster': clusters }
frame = pd.DataFrame(doc, index = [clusters] , columns = ['title', 'synopsis','cluster'])
print(frame['cluster'].value_counts())
print(frame)
#frame = pd.DataFrame(doc, index = [clusters] , columns = ['title', 'cluster'])
#print('done1')

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title']:
        print(title, end=' ')
    print() #add whitespace
    print() #add whitespace'''
    
    
print()
print()