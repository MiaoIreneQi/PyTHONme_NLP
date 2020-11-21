from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for sentence in twitter:
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
    

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

tempo_tweet = tweet_complete.groupby('Date')['tweet'].apply(list)
tempo_tweet.tolist()
date_distinct_tweet = tempo_tweet.tolist()
date_distinct_continuous_tweet = []
for element in date_distinct_tweet:
    date_distinct_continuous_tweet.append('\n\n'.join(element))
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_tweet = SentimentIntensityAnalyzer()
score_list_tweet = []
for tweet in date_distinct_continuous_tweet:
    score_list_tweet.append(sid_tweet.polarity_scores(tweet))
    

score_df_tweet = pd.DataFrame()

for element in score_list_tweet:
    score_df_tweet = score_df_tweet.append(element, ignore_index = True)
distinct_time_tweet = sorted(pd.to_datetime(list(set(tweet_complete['Date']))))

score_df_tweet.insert(loc = 0, column = 'Date', value = distinct_time_tweet)



score_df_tweet.set_index('Date', inplace = True)



t_index = pd.date_range('2017-01-03','2020-11-13')
score_df2_tweet = score_df_tweet.reindex(t_index, fill_value = nan)
score_df2_tweet.index = [dt.date() for dt in score_df2_tweet.index]
