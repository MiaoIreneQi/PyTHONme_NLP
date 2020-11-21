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




    
    
    
tempo_script = table_for_all_articles.groupby('Date')['Article continuous'].apply(list)
tempo_script.tolist()
date_distinct_script = tempo_script.tolist()
    
date_distinct_continuous_script = []

for element in date_distinct_script:
    date_distinct_continuous_script.append('\n\n'.join(element))
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_script = SentimentIntensityAnalyzer()
score_list_script = []
for article in date_distinct_continuous_script:
    score_list_script.append(sid_script.polarity_scores(article))
 
score_df_script = pd.DataFrame()
for element in score_list_script:
    score_df_script = score_df_script.append(element, ignore_index = True)
distinct_time = sorted(pd.to_datetime(list(set(table_for_all_articles['Date']))))[::-1]

score_df_script.insert(loc = 0, column = 'Date', value = distinct_time)
score_df_script['Date'] = pd.to_datetime(score_df_script['Date'])
score_df_script.set_index('Date', inplace = True)
t_index = pd.date_range('2017-01-03','2020-11-13')
score_df2_script = score_df_script.reindex(t_index, fill_value = 0)
score_df2_script.index = [dt.date() for dt in score_df2_script.index]


sp_500 = pd.read_excel('S&P 500 Dataset.xlsx')
holiday_list = []
for date in t_index:
    if date not in sp_500['date'].to_list():
        holiday_list.append(date)

tweet1 = pd.read_csv('trump_20200530_clean.csv')
tweet1.rename(columns = {'datetime' : 'Date'}, inplace = True)
tweet1.Date = pd.to_datetime(tweet1.Date)

new_tweet = pd.read_csv('new twitter.csv')
new_tweet_list = new_tweet['text,created_at'].to_list()
new_tweet_date = [element[-19:-9] for element in new_tweet_list]
new_tweet.insert(loc = 0, column = 'Date', value = pd.to_datetime(new_tweet_date))
new_tweet.rename(columns = {'text,created_at' : 'tweet'}, inplace = True)

tweet_to_may_30 = tweet1[['Date', 'tweet']]
tweet_complete = pd.concat([new_tweet,tweet_to_may_30], ignore_index= True)
tweet_complete.sort_values(by = 'Date', ignore_index = True, inplace = True)
tweet_complete.Date = [dt.date() for dt in tweet_complete.Date]
