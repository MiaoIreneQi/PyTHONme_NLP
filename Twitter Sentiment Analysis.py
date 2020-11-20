###twitter sentiment analysis
import pandas as pd
import numpy as np
import csv

#csv to list 
twitter = []
with open('/Users/irene/Desktop/FINA4350/group project code/into-heart-of-darkness-master/trump_20200530.csv') as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    for row in data_reader:
        twitter.append(row[1])
#csv to 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for sentence in twitter:
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
    
