#open csv file
import pandas as pd
import os
os.chdir('/Users/jiangbinghan/Desktop/Yr4 S1/NLP in Finance and Fintech/github_jbh/PyTHONme_NLP')
df = pd.read_csv('table for all articles_2.csv')
df = pd.read_csv('article.csv')

a =[]
a = df['0'].to_list()
a[-1]
a

#extract articles in the csv file
#import csv

#with open('table for all articles_2.csv') as csvfile:
#    data_reader = csv.reader(csvfile, delimiter=',')
#    for row in data_reader:
#        print(row[2])

#csv extract --> string
#articles_str=''
#with open('article.csv') as csvfile:
#    data_reader = csv.reader(csvfile, delimiter=',')
#    for row in data_reader:
#        articles_str.join(row[2])
#print(articles_str)

#articles #turn them into lists (however, the file is too big)
#articles[-1]

#turn them into strings-DONE
article_string=''
article_string = article_string.join(articles)

#turn strings into a text file -DONE
#text_file = open('data-gensim-mycorpus.txt', 'w')
#n = text_file.write(article_string)
#text_file.close

#streaming corpus - not sure what happend here
#class MyCorpus(object):   # Class that instantiates an iterable.
#    def __iter__(self):   # Define a generator function.
#        for line in open('data-gensim-mycorpus.txt'):
#            yield line.lower().split()


#sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_twitter = SentimentIntensityAnalyzer()
sid_twitter.polarity_scores(article_string)




    
    
    
tempo = table_for_all_articles.groupby('Date')['Article continuous'].apply(list)
tempo.tolist()
date_distinct = tempo.tolist()
    
date_distinct_continuous = []

for element in date_distinct:
    date_distinct_continuous.append('\n\n'.join(element))
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_script = SentimentIntensityAnalyzer()
score_list = []
for article in date_distinct_continuous:
    score_list.append(sid_script.polarity_scores(article))
 
score_df = pd.DataFrame()
for element in score_list:
    score_df = score_df.append(element, ignore_index = True)

score_df.insert(loc = 0, column = 'Date', value = distinct_time)
score_df.set_index('Date', inplace = True)
t_index = pd.date_range('2017-01-03','2020-11-13')
score_df2 = score_df.reindex(t_index, fill_value = 0)

sp_500 = pd.read_excel('S&P 500 Dataset.xlsx')
holiday_list = []
for date in t_index:
    if date not in sp_500['date'].to_list():
        holiday_list.append(date)
