#open csv file
import pandas as pd
import os
os.chdir('/Users/jiangbinghan/Desktop/Yr4 S1/NLP in Finance and Fintech/github_jbh/PyTHONme_NLP')
df = pd.read_csv('table for all articles_2.csv')

#extract articles in the csv file
import csv

with open('table for all articles_2.csv') as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    for row in data_reader:
        print(row[2])



articles #turn them into lists (however, the file is too big)
articles[-1]

#turn them into strings-DONE
article_string=''
article_string = article_string.join(articles)


#turn strings into a text file -DONE
text_file = open('data-gensim-mycorpus.txt', 'w')
n = text_file.write(article_string)
text_file.close

#streaming corpus - not sure what happend here
class MyCorpus(object):   # Class that instantiates an iterable.
    def __iter__(self):   # Define a generator function.
        for line in open('data-gensim-mycorpus.txt'):
            yield line.lower().split()


#sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_twitter = SentimentIntensityAnalyzer()
sid_twitter.polarity_scores(article_string)
