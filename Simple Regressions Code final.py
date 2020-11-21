# Group: PyTHONme_NLP
# Project topic: Does Donald Trump's speech pose significant influence on the U.S. stock market?
# Authors in collaboration:
#QI Miao Irene, 3035448988
#CHEN Jingshu David,
#JIANG Binghan Stephanie, 3035447180
#BAO Enqi Bruce,


#Object1: web scrapping from webpage: https://www.rev.com/blog/transcript-category/donald-trump-transcripts

#to start with, import packages needed.
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


#################################################

# Step 1: obtain the titles in Page 1

r = \
    requests.get(
        'https://www.rev.com/blog/transcript-category/donald-trump-transcripts?view=all', timeout=5)

clean_transcript_p1 = BeautifulSoup(r.text, 'lxml')
href_list_page1 = [tag.get('href') for tag in clean_transcript_p1.find_all('a')]
transcript_href_list_page1 = [element for element in href_list_page1 if 'https://www.rev.com/blog/transcripts/' in element]

tag_name_list = {tag.name for tag in clean_transcript_p1.find_all(True)}

title_p1 = [tag.text for tag in clean_transcript_p1.find_all(['strong'])]
title_p1.remove("Help Us Improve the Rev Transcript Library!")



# Step 2.1 Creating https for turining the pages (from page 2 onwards)
address = "https://www.rev.com/blog/transcript-category/donald-trump-transcripts/page/{}?view=all"
last_page_candidates = [tag.text for tag in clean_transcript_p1.find_all('a', {'class' : 'page-numbers'})]

last_page = int(last_page_candidates[-2])

web_list = []
for i in range (2, last_page + 1):
    web_list.append(address.format(i))



# Step 2.2 Obtaining the titles in from Page 2 onwards

title_list = []
transcirpt_href_list_from_p2 = []
for web in web_list:
    r = requests.get(web, timeout=5)
    clean_transcript = BeautifulSoup(r.text, 'lxml')
    title_list.append([tag.text for tag in clean_transcript.find_all('strong')])
    href_list_web = [tag.get('href') for tag in clean_transcript.find_all('a')]
    transcirpt_href_list_from_p2.append([element for element in href_list_web if 'https://www.rev.com/blog/transcripts/' in element])

    # remove the unnecessary titles from the list (loop):
for sublist in title_list:
    sublist.remove("Help Us Improve the Rev Transcript Library!")

    # from list in list to one list.
title_list_unnested = [item for sublist in title_list for item in sublist]
transcirpt_href_list_from_p2_unnested = [item for sublist in transcirpt_href_list_from_p2 for item in sublist]



# Step 2.3 Combine the title lists of page 1 and pages from page 2 onwards

title_list_unnested = title_p1 + title_list_unnested
transcirpt_href_list_unnested = transcript_href_list_page1 + transcirpt_href_list_from_p2_unnested



#Get all the article from href
articles = []
articles_in_paragraph = []
for href in transcirpt_href_list_unnested:
    article_raw = requests.get(href).text
    article_s = BeautifulSoup(article_raw, 'lxml')
    cleaned_article_in_paragraph = [tag.text for tag in article_s.find_all('p')]
    #Delelte unnecesary content if any
    if 'Transcribe Your Own Content' in cleaned_article_in_paragraph:
        cleaned_article_in_paragraph.remove('Transcribe Your Own Content')
    if ' Try Rev and save time transcribing, captioning, and subtitling.' in cleaned_article_in_paragraph:
        cleaned_article_in_paragraph.remove\
        (' Try Rev and save time transcribing, captioning, and subtitling.')
    cleaned_article_in_paragraph.pop(-1)
    articles_in_paragraph.append(cleaned_article_in_paragraph)
    article = '\n'.join(cleaned_article_in_paragraph)
    articles.append(article)

date = [sublist.pop(0) for sublist in articles_in_paragraph]

table_for_all_articles = pd.DataFrame({'Title': title_list_unnested,
                                       'Date': date,
                                       'Article in paragraphs': articles_in_paragraph,
                                       'Article continuous': articles})
#Set Title column as the index column
table_for_all_articles = table_for_all_articles.set_index('Title')



#Preprocessing
tokenize_list = [regexp_tokenize(article, r'\w+')
                 for article in table_for_all_articles\
                     ['Article continuous']]

no_stops_collection =\
[[t for t in article if t.lower() not in stopwords.words('english')]
 for article in tokenize_list]

no_numeral = [[t for t in article if not t.isnumeric()]
              for article in no_stops_collection]

#wnl = WordNetLemmatizer()

#lemmatized_counting = [Counter(article) for article in lemmatized]

#ngs = [ngrams(element, 2) for element in no_stops_collection]
#gram_2_list = [[' '.join(ng) for ng in element] for element in ngs]
#counting_gram_2 = [Counter(article) for article in gram_2_list]
###################################################################################################

#object 2: Transcript Sentiment Analysis

#open csv file
import pandas as pd
import os
import csv
import pickle
#change the working directory on your computer
os.chdir('yourpath')

#open pickle file
with open('table for all articles.pickle', 'rb') as f:
    table_for_all_articles = pickle.load(f)

#Group transcripts that are released on the same date:
tempo_script = table_for_all_articles.groupby('Date')['Article continuous'].apply(list)
date_distinct_script = tempo_script.tolist() #convert to list

#Add missing dates into the list to match the date series of S&P 500 data file.
date_distinct_continuous_script = []

for element in date_distinct_script:
    date_distinct_continuous_script.append('\n\n'.join(element))

#Sentiment Analysis - transcript
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid_script = SentimentIntensityAnalyzer()
score_list_script = []
for article in date_distinct_continuous_script:
    score_list_script.append(sid_script.polarity_scores(article))

#Save transcript sentiment scores to a DataFrame
score_df_script = pd.DataFrame()
for element in score_list_script:
    score_df_script = score_df_script.append(element, ignore_index = True)

import datetime
#pick out the distinct date for transcripts
#transform the data format for date into datetime format
#sort the transcript date
distinct_time = sorted(pd.to_datetime(list(set(table_for_all_articles['Date']))))[::-1]

#insert distinct time for transcipts to the transcript sentiment scores dataframe.
score_df_script.insert(loc = 0, column = 'Date', value = distinct_time)

#transform the data format for date into datetime format
score_df_script['Date'] = pd.to_datetime(score_df_script['Date'])

#use reindex to add the missing calendar dates into the transcript sentiment score dataframe
score_df_script.set_index('Date', inplace = True)
t_index = pd.date_range('2017-01-03','2020-11-13') #set a index with all of the calendar dates we need

#fill in the value as nan (from numpy)
import numpy as np
from numpy as nan
score_df2_script = score_df_script.reindex(t_index, fill_value = nan)
score_df2_script.index = [dt.date() for dt in score_df2_script.index] #extract the date only (exclude the hour, minutes, seconds, microseconds)
score_df2_script.reset_index(inplace = True)
score_df2_script.rename(columns = {'index' : 'Date'}, inplace = True) #rename the columns for future reference
score_df2_script.Date = pd.to_datetime(score_df2_script.Date)

sp_500 = pd.read_excel('S&P 500 Dataset.xlsx')
#extract the holiday dates on which the market is closed (and thus there is no S&P500 data)
holiday_list = []
for date in t_index:
    if date not in sp_500['date'].to_list():
        holiday_list.append(date)

#######################################################################################


#Sentiment Analysis: Tweets

#Import tweets from 2017/01/03 to 2020/05/30 into a dataframe
import os
os.chdir('yourpath')

tweet1 = pd.read_csv('trump_20200530_clean.csv')
tweet1.rename(columns = {'datetime' : 'Date'}, inplace = True) #rename the columns for future reference
tweet1.Date = pd.to_datetime(tweet1.Date) #transform the data format into datetime

#Import tweets from 2020/05/30 to 2020/11/13 into another dataframe
new_tweet = pd.read_csv('new twitter.csv')
new_tweet_list = new_tweet['text,created_at'].to_list() #transform the dataframe into a list

#Because the date was in the content of the new tweets, we need to extract the dates.
new_tweet_date = [element[-19:-9] for element in new_tweet_list] #extract the date from the list
new_tweet.insert(loc = 0, column = 'Date', value = pd.to_datetime(new_tweet_date)) #insert date of tweets into the dataframe
new_tweet.rename(columns = {'text,created_at' : 'tweet'}, inplace = True)

#extract only the date and the tweet from the old tweets (from 2017/01/03 to 2020/05/30)
#to be accords with the new tweets(rom 2020/05/30 to 2020/11/13).
tweet_to_may_30 = tweet1[['Date', 'tweet']]

#Concatenate the old tweets with the new tweets
tweet_complete = pd.concat([new_tweet,tweet_to_may_30], ignore_index= True)
tweet_complete.sort_values(by = 'Date', ignore_index = True, inplace = True)
tweet_complete.Date = [dt.date() for dt in tweet_complete.Date] #extract the date only (exclude the hour, minutes, seconds, microseconds)

#Group tweets that are released on the same date:
tempo_tweet = tweet_complete.groupby('Date')['tweet'].apply(list)
tempo_tweet.tolist()

#Add missing dates into the list to match the date series of S&P 500 data file.
date_distinct_tweet = tempo_tweet.tolist()
date_distinct_continuous_tweet = []
for element in date_distinct_tweet:
    date_distinct_continuous_tweet.append('\n\n'.join(element))

#Perform sentimate analysis to tweets
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

#Use reindex to add in missing calendar date
#fill value to nan
import numpy as np
from numpy import nan
t_index = pd.date_range('2017-01-03','2020-11-13')
score_df2_tweet = score_df_tweet.reindex(t_index, fill_value = nan)
score_df2_tweet.index = [dt.date() for dt in score_df2_tweet.index]

score_df2_tweet.reset_index(inplace = True)
score_df2_tweet.rename(columns = {'index' : 'Date'}, inplace = True) #rename the columns for future reference
score_df2_tweet.Date = pd.to_datetime(score_df2_tweet.Date)
##################################################################################

#RUN REGRESSIONS

#Step1: import data (dependent variable)
#Import the S&P500 data into a dataframe
sp_500 = pd.read_excel('S&P 500 Dataset.xlsx')
sp_500.rename(columns = {'date' : 'Date'}, inplace = True)
sp_500.Date = pd.to_datetime(sp_500.Date)

#Import Volatility Index into a dataframe
vix = pd.read_excel('VIX.xlsx')
vix.rename(columns = {'date' : 'Date'}, inplace = True)
vix.Date = pd.to_datetime(vix.Date)

#Import industry indices into a dataframe
data_new = pd.read_excel('data_new.xlsx')
data_new.index.name = 'index'
data_new.Date = pd.to_datetime(data_new.Date)

#Step2: Merge DataFrame
#Step 2.1 Menge sentiment scores together
sentiment_combined = pd.merge(score_df2_script, score_df2_tweet, on = 'Date')
#Step 2.2 Merge sentiment scores with their absolute values
#create columns to compute the absolute values of the compound sentiment scores for transcripts and tweets
sentiment_combined.insert(loc=0, column = 'absolute_compound_script', value = sentiment_combined['compound_x'])
sentiment_combined['absolute_compound_script'] = sentiment_combined['compound_x'].abs()

sentiment_combined.insert(loc=0, column = 'absolute_compound_tweet', value = sentiment_combined['compound_y'])
sentiment_combined['absolute_compound_tweet'] = sentiment_combined['compound_y'].abs()

#Step 2.3 Merge sentiment scores with S&P500 dataframe
analysis_sp500 = pd.merge(sentiment_combined, sp_500, on = 'Date')
#Step 2.4 Merge analysis_sp500 with other industry indices dataframe
analysis_all = pd.merge(analysis_sp500, data_new, on = 'Date')

analysis_all.rename(columns = {'S&P 500 index':'SP_500_index','percentage change' : 'percentage_change'}, inplace = True)

#Step3: instantiate variables for plotting
import matplotlib.pyplot as plt
y_1 = analysis_all['SP_500_index']
y_2 = analysis_all['percentage_change']
y_3 = analysis_all['index']
y_4 = analysis_all['pc_DJ_industrial']
y_5 = analysis_all['pc_Dj_utility']
y_6 = analysis_all['pc_WS_housing']
y_7 = analysis_all['Energy_ETFVIX']

x_1 = analysis_all['absolute_compound_script']
x_2 = analysis_all['absolute_compound_tweet']
x_3 = analysis_all['compound_script']
x_4 = analysis_all['compound_tweet']
x_5 = analysis_all['Interest_rate']

#Step4: REGRESSIONS: market index (percentage change)
#Step4.1 regression: market index (percentage change) versus the sentiment scores(compound)
import statsmodels.formula.api as sm

result = sm.ols(formula="percentage_change ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step4.2 regression: market index (percentage change) versus the sentiment scores(absolute)
result = sm.ols(formula="percentage_change ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step4.3 plots
plt.scatter(x_4, y_2) #x: compound_tweet y: percentage_change
plt.ylabel('S&P500 %')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 1.1')
plt.show()

plt.scatter(x_2, y_2)
plt.ylabel('S&P500 %')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 1.2')
plt.show()

#Step5: REGRESSIONS: volatility index
#Step5.1 regression: volatility index versus the sentiment scores(compound)
result = sm.ols(formula="index ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())
#Step5.2 regression: volatility index versus the sentiment scores(absolute value)
result = sm.ols(formula="index ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step5.3 plots
plt.scatter(x_4, y_3)
plt.ylabel('Volatility Index')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 2.1')
plt.show()

plt.scatter(x_2, y_3)
plt.ylabel('Volatility Index')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 2.2')
plt.show()

#Step6: REGRESSIONS: Dow & Jones Industrial Average Index
#Step6.1 regression: Dow & Jones Industrial Average Index versus the sentiment scores(compound)
result = sm.ols(formula="pc_DJ_industrial ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step6.2 regression: Dow & Jones Industrial Average Index versus the sentiment scores(absolute value)
result = sm.ols(formula="pc_DJ_industrial ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step6.3 plots
plt.scatter(x_4, y_4)
plt.ylabel('DJ_industrial %')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 3.1')
plt.show()

plt.scatter(x_2, y_4)
plt.ylabel('DJ_industrial %')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 3.2')
plt.show()

#Step7: REGRESSIONS: Dow & Jones Utility Average
#Step7.1 regression: Dow & Jones Utility Average versus the sentiment scores(compound)
result = sm.ols(formula="pc_Dj_utility ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step7.1 regression: Dow & Jones Utiltiy Average versus the sentiment scores(absolute value)
result = sm.ols(formula="pc_Dj_utility ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())
#########################
#Step7.3 plots
plt.scatter(x_4, y_5)
plt.ylabel('DJ_utility %')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 4.1')
plt.show()

plt.scatter(x_2, y_5)
plt.ylabel('DJ_utility %')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 4.2')
plt.show()

#Step8: REGRESSIONS: Wilshire real estate index
#Step8.1 regression: Wilshire real estate index versus the sentiment scores(compound)
result = sm.ols(formula="pc_WS_housing ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step8.2 regression: Wilshire real estate index versus the sentiment scores(absolute value)
result = sm.ols(formula="pc_WS_housing ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step8.3 plots
plt.scatter(x_4, y_6)
plt.ylabel('WS_housing %')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 5.1')
plt.show()

plt.scatter(x_2, y_6)
plt.ylabel('WS_housing %')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 5.2')
plt.show()


#Step9: REGRESSIONS: Energy ETF volatility index
#Step9.1 regression: Energy ETF volatility index versus the sentiment scores(compound)
result = sm.ols(formula="Energy_ETFVIX ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step9.2 regression: Energy ETF volatility index versus the sentiment scores(absolute value)
result = sm.ols(formula="Energy_ETFVIX ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
print(result.summary())

#Step9.3 plots
plt.scatter(x_4, y_6)
plt.ylabel('Energy_ETFVIX')
plt.xlabel('tweet_sentiment_score')
plt.title('regression 6.1')
plt.show()

plt.scatter(x_2, y_6)
plt.ylabel('Energy_ETFVIX')
plt.xlabel('tweet_sentiment_score(abs)')
plt.title('regression 6.2')
plt.show()

#THE END OF THE REGRESSION
###################################################################################################
