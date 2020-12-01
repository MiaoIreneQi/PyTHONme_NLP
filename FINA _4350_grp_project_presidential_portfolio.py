# Group: PyTHONme_NLP
# Project topic: presidential portfolio construction
# Authors in collaboration: QI Miao Irene, CHEN Jingshu David, JIANG Binghan Stephanie, BAO Enqi Bruce


#to start with, import packages needed. 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from numpy import nan
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statsmodels.formula.api as sm
import statsmodels.api as sm1
import matplotlib.pyplot as plt
  

  
######Object1: main branch - web scrapping from webpage: https://www.rev.com/blog/transcript-category/donald-trump-transcripts#########
r = requests.get('https://www.rev.com/blog/transcript-category/donald-trump-transcripts?view=all', timeout=5)
    
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
transcript_href_list_from_p2 = []                  
for web in web_list:
    r = requests.get(web, timeout=5)
    clean_transcript = BeautifulSoup(r.text, 'lxml')
    title_list.append([tag.text for tag in clean_transcript.find_all('strong')])
    href_list_web = [tag.get('href') for tag in clean_transcript.find_all('a')]
    transcript_href_list_from_p2.append([element for element in href_list_web if 'https://www.rev.com/blog/transcripts/' in element])

# remove the unnecessary titles from the list (loop): 
for sublist in title_list:
    sublist.remove("Help Us Improve the Rev Transcript Library!")

# from nested list in list to a single list.
title_list_unnested = [item for sublist in title_list for item in sublist]
transcript_href_list_from_p2_unnested = [item for sublist in transcript_href_list_from_p2 for item in sublist]



# Step 2.3 Combine the title lists of page 1 and pages from page 2 onwards

title_list_unnested = title_p1 + title_list_unnested
transcript_href_list_unnested = transcript_href_list_page1 + transcript_href_list_from_p2_unnested



#Get all the article from href                
articles = []
articles_in_paragraph = []
for href in transcript_href_list_unnested:
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


################################Sentiment score of speech transcripts by date####################################

#group transcripts by date so that if there multiple transcripts on a single date, they will be combined as one.
tempo_script = table_for_all_articles.groupby('Date')['Article continuous'].apply(list)
tempo_script.tolist()
date_distinct_script = tempo_script.tolist()
    
date_distinct_continuous_script = []

for element in date_distinct_script:
    date_distinct_continuous_script.append('\n\n'.join(element))

#Getting sentiment scores
sid_script = SentimentIntensityAnalyzer()
score_list_script = []
for article in date_distinct_continuous_script:
    score_list_script.append(sid_script.polarity_scores(article))
 
#Create dataframes
score_df_script = pd.DataFrame()
for element in score_list_script:
    score_df_script = score_df_script.append(element, ignore_index = True)
distinct_time = sorted(pd.to_datetime(list(set(table_for_all_articles['Date']))))[::-1]

score_df_script.insert(loc = 0, column = 'Date', value = distinct_time)
score_df_script['Date'] = pd.to_datetime(score_df_script['Date'])
score_df_script.set_index('Date', inplace = True)
t_index = pd.date_range('2017-01-03','2020-11-13')
score_df2_script = score_df_script.reindex(t_index, fill_value = nan)
score_df2_script.index = [dt.date() for dt in score_df2_script.index]
score_df2_script.reset_index(inplace = True)
score_df2_script.rename(columns = {'index' : 'Date'}, inplace = True)



#########################################Getting tweets##############################################

#Note that the tweet files were prepared by Stephanie and Irene

#read tweets from Jan 3, 2017 to May 30, 2020
tweet1 = pd.read_csv('trump_20200530_clean.csv')
tweet1.rename(columns = {'datetime' : 'Date'}, inplace = True)
tweet1.Date = pd.to_datetime(tweet1.Date)

#read tweets from May 30 onwayds
new_tweet = pd.read_csv('new twitter.csv')
new_tweet_list = new_tweet['text,created_at'].to_list()
new_tweet_date = [element[-19:-9] for element in new_tweet_list]
new_tweet.insert(loc = 0, column = 'Date', value = pd.to_datetime(new_tweet_date))
new_tweet.rename(columns = {'text,created_at' : 'tweet'}, inplace = True)

tweet_to_may_30 = tweet1[['Date', 'tweet']]
tweet_complete = pd.concat([new_tweet,tweet_to_may_30], ignore_index= True)
tweet_complete.sort_values(by = 'Date', ignore_index = True, inplace = True)
tweet_complete.Date = [dt.date() for dt in tweet_complete.Date]

###########################################Sentiment score of tweets by date########################################
#group tweets by date
tempo_tweet = tweet_complete.groupby('Date')['tweet'].apply(list)
tempo_tweet.tolist()
date_distinct_tweet = tempo_tweet.tolist()
date_distinct_continuous_tweet = []
for element in date_distinct_tweet:
    date_distinct_continuous_tweet.append('\n\n'.join(element))

#Getting sentiment scores
sid_tweet = SentimentIntensityAnalyzer()
score_list_tweet = []
for tweet in date_distinct_continuous_tweet:
    score_list_tweet.append(sid_tweet.polarity_scores(tweet))
    

#Create dataframes
score_df_tweet = pd.DataFrame()

for element in score_list_tweet:
    score_df_tweet = score_df_tweet.append(element, ignore_index = True)
distinct_time_tweet = sorted(pd.to_datetime(list(set(tweet_complete['Date']))))

score_df_tweet.insert(loc = 0, column = 'Date', value = distinct_time_tweet)



score_df_tweet.set_index('Date', inplace = True)

#reindex
t_index = pd.date_range('2017-01-03','2020-11-13')

#fill value to nan 
score_df2_tweet = score_df_tweet.reindex(t_index, fill_value = nan)
score_df2_tweet.index = [dt.date() for dt in score_df2_tweet.index]
score_df2_tweet.reset_index(inplace = True)
score_df2_tweet.rename(columns = {'index' : 'Date'}, inplace = True)




####################################Merge dataframes for sentiment scores of speech transcripts and tweets#############
sentiment_combined = pd.merge(score_df2_script, score_df2_tweet, on = 'Date', 
                              suffixes = ('_script','_tweet'))


#########################Import S&P 500 and Volatility Index, both prepared by Irene and Stephanie##########################
sp_500 = pd.read_excel('S&P 500 Dataset.xlsx')
sp_500.rename(columns = {'date' : 'Date', 'percentage change' : 'pc_sp_500'}, inplace = True)
sp_500.Date = [dt.date() for dt in sp_500.Date]

vix = pd.read_excel('VIX.xlsx', sheet_name = 'Sheet1')
vix.rename(columns = {'date' : 'Date', 'index' : 'VIX'}, inplace = True)
vix.Date = pd.to_datetime(vix.Date)

#merge with sentiment_combined
analysis_all = pd.merge(sp_500, sentiment_combined, on = 'Date')
analysis_all.Date = pd.to_datetime(analysis_all.Date)
analysis_all = pd.merge(analysis_all, vix, how = 'left')

#########################import some indexes in data_new, prepared by Irene and Stephanie#########################
data_new = pd.read_excel('data_new.xlsx')
data_new.replace([0], nan, inplace = True)

#Preparing data_new for merging
data_new.Date = pd.to_datetime(data_new.Date)
data_new.set_index('Date', inplace = True)
data_new2 = data_new.reindex(t_index, fill_value = nan)
data_new2.reset_index(inplace = True)
data_new2.rename(columns = {'index' : 'Date'}, inplace = True)
                         
#merge into analysis_all
analysis_all = pd.merge(analysis_all, data_new2, on = 'Date')


##############################General Regression by Stephanie, David, Irene############################################
#Get some aboslute values for regressions later
analysis_all['absolute_compound_script'] = analysis_all['compound_script'].abs()
analysis_all['absolute_compound_tweet'] = analysis_all['compound_tweet'].abs()


y_2 = analysis_all['pc_sp_500']
y_3 = analysis_all['VIX']
y_4 = analysis_all['pc_DJ_industrial']
y_5 = analysis_all['pc_Dj_utility']
y_6 = analysis_all['pc_WS_housing']
y_7 = analysis_all['Energy_ETFVIX']

x_1 = analysis_all['absolute_compound_script']
x_2 = analysis_all['absolute_compound_tweet']
x_3 = analysis_all['compound_script']
x_4 = analysis_all['compound_tweet']
x_5 = analysis_all['Interest_rate']
x_6 = analysis_all['pos_script']
x_7 = analysis_all['neg_script']
x_8 = analysis_all['pos_tweet']
x_9 = analysis_all['neg_tweet']

result = sm.ols(formula="pc_sp_500 ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

result = sm.ols(formula="pc_sp_500 ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
print(result.summary())

#Step4.2 regression: market index (pc_sp_500) versus the sentiment scores(absolute)
result = sm.ols(formula="pc_sp_500 ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
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
result = sm.ols(formula="VIX ~ compound_script + compound_tweet", data=analysis_all).fit()
print(result.summary())

result = sm.ols(formula="VIX ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
print(result.summary())

#Step5.2 regression: volatility index versus the sentiment scores(absolute value)
result = sm.ols(formula="VIX ~ absolute_compound_script + absolute_compound_tweet", data=analysis_all).fit()
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

result = sm.ols(formula="pc_DJ_industrial ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
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

result = sm.ols(formula="pc_Dj_utility ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
print(result.summary())

#Step7.2 regression: Dow & Jones Utiltiy Average versus the sentiment scores(absolute value)
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

result = sm.ols(formula="pc_WS_housing ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
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

result = sm.ols(formula="Energy_ETFVIX ~ compound_script + compound_tweet + Interest_rate", data=analysis_all).fit() #plus interest rate
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

#Step 10 Regression: positive, negative sentiment scores
result = sm.ols(formula="pc_sp_500 ~ pos_script + neg_script + pos_tweet + neg_tweet + Interest_rate", data=analysis_all).fit()
print(result.summary())

#plt.scatter(x_6, y_2)
#plt.scatter(x_7, y_2)
#plt.scatter(x_8, y_2)
#plt.scatter(x_9, y_2) 
#THE END OF THE Genral REGRESSION


##########Tweet sentiment (compound) score aboslute value intervals & correpsonding Standard Deviations of Daily S&P 500 Percentage Changes#####
#Data Preparation
interval = np.linspace(0,1,101)
k = analysis_all['absolute_compound_tweet']
midpoint = np.linspace(0.005, 0.995, 100)
std_list = []
for i in range(100):
    std = analysis_all[(k>interval[i])&(k<interval[i+1])]['pc_sp_500'].std()
    std_list.append(std)
tweet_interval_std_df = pd.DataFrame({'midpoint_of_absolute_compound_tweet_interval' : midpoint,
                                      'std_by_interval_absolute_compound_tweet' : std_list})

#Regression
result = sm.ols(formula="std_by_interval_absolute_compound_tweet ~ midpoint_of_absolute_compound_tweet_interval", data=tweet_interval_std_df).fit()
print(result.summary())

#Preparation for plot (credit to StackOverflow)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Plot
plt.scatter(tweet_interval_std_df['midpoint_of_absolute_compound_tweet_interval'],
tweet_interval_std_df['std_by_interval_absolute_compound_tweet'])
plt.xlabel('Midpoint of Absolute Value Compound Score Interval')
plt.ylabel('Corresponding Standard Deviation of Daily Percentage Changes of S&P 500')
plt.title('regression of std of %S&P 500 on midpoint of intervals of tweet sentiment')
plt.show()


############################Create year dummy variable for dummy year wise regressions##############################
def dummy_year(year):
    dummy = []
    for date in analysis_all['Date'].tolist():
        if date.year == year:
            dummy.append(1)
        else:
            dummy.append(0)
    return dummy

analysis_all['dummy17'] = dummy_year(2017)

analysis_all['dummy18'] = dummy_year(2018)

analysis_all['dummy19'] = dummy_year(2019)

analysis_all['dummy20'] = dummy_year(2020)



########################################Year wise regression of DJ industrial on sentiment scores#######################################
print()
print('run DJ_industry on compund_tweet if year ==2017: ' +\
      'we found significantly positive result! Although the magnitude is very small')
result = sm.ols(formula="pc_DJ_industrial ~ compound_tweet", data=analysis_all[analysis_all['dummy17'] == 1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'compound_tweet',fig=fig)
print()

print('run DJ_industry on compund_tweet if year ==2018: ' +\
      'we found significantly negative result! Although the magnitude is very small')
result = sm.ols(formula="pc_DJ_industrial ~ compound_tweet", data=analysis_all[analysis_all['dummy18'] == 1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'compound_tweet',fig=fig)
print()

print('run DJ_industry on compund_tweet if year ==2019: ' +\
      'we found insignificant result')
result = sm.ols(formula="pc_DJ_industrial ~ compound_tweet", data=analysis_all[analysis_all['dummy19'] == 1]).fit()
print(result.summary())
print()

print('run DJ_industry on compund_tweet if year ==2020: ' +\
      'we found insignificant result)
result = sm.ols(formula="pc_DJ_industrial ~ compound_tweet", data=analysis_all[analysis_all['dummy20'] == 1]).fit()
print(result.summary())
print()





################################Analyze (mannually) selected keywords in speech transcripts####################################
keywords = ['China', 'tariff', 'Xi', 'Putin', 'tax', 
            'COVID', 'virus', 'fake', 'abortion', 'Russia']


def word_count(word):
    counting = []
    for script in date_distinct_continuous_script:
        counting.append(script.count(word))
    return counting

keyword_dic = {}
for keyword in keywords:
    keyword_dic[keyword] = word_count(keyword)

keyword_df = pd.DataFrame({keyword : keyword_dic[keyword] for keyword in keywords},
                         index = sorted(list(set(table_for_all_articles.Date))))

keyword_df.index = pd.to_datetime(keyword_df.index)

keyword_df2 = keyword_df.reindex(t_index, fill_value = nan)

keyword_df2.reset_index(inplace = True)
keyword_df2.rename(columns = {'index' : 'Date'}, inplace = True)

##########################################Analyze (mannually) selected keywords in tweets################################################
# Recall taht keywords = ['China', 'tariff', 'Xi', 'Putin', 'tax', 
            #'COVID', 'virus', 'fake', 'abortion', 'Russia']

def tweet_word_counting(word):
    counting = []
    for tweet in date_distinct_continuous_tweet:
        counting.append(tweet.count(word))
    return counting

keyword_dic_tweet = {}
for keyword in keywords:
    keyword_dic_tweet[keyword] = tweet_word_counting(keyword)

keyword_tweet_df = pd.DataFrame({keyword : keyword_dic_tweet[keyword]
                                 for keyword in keywords}, 
                                index = sorted(list(set(tweet_complete.Date))))
keyword_tweet_df.index = pd.to_datetime(keyword_tweet_df.index)

keyword_tweet_df2 = keyword_tweet_df.reindex(t_index, fill_value = nan)
keyword_tweet_df2.reset_index(inplace = True)
keyword_tweet_df2.rename(columns = {'index' : 'Date'}, inplace = True)

###################################Merge dataframes of keywords for both script and tweets############################################
keyword_analysis = pd.merge(keyword_df2, keyword_tweet_df2, on = 'Date', 
                            suffixes = ('_script', '_tweet'))

analysis_all = pd.merge(analysis_all, keyword_analysis, on = 'Date')

#######################################Combine word counts of 'China' and 'Xi' and those of 'Russia' and 'Putin
analysis_all['China_compound_script'] = analysis_all['China_script'] +\
    analysis_all['Xi_script']

analysis_all['China_compound_tweet'] = analysis_all['China_tweet'] +\
    analysis_all['Xi_tweet']

analysis_all['Russia_compound_script'] = analysis_all['Russia_script'] +\
    analysis_all['Putin_script']

analysis_all['Russia_compound_tweet'] = analysis_all['Russia_tweet'] +\
    analysis_all['Putin_tweet']


############################################Regressions on keyword counting#################################################
print('Regress pc_DJ_industrial on number of keyword "China" in tweets and script: '+/
      'significantly negative for tweet, insignificant for script')
result = sm.ols(formula="pc_DJ_industrial ~ China_compound_tweet + China_compound_script", data=analysis_all).fit()
print(result.summary())
print()

print('Regress pc_DJ_industrial on number of keyword "China" in tweets and script' +/
      'in year 2019: significantly negative for tweet, significantly positive for' +/
      'script (smaller magnitude than tweet)')
result = sm.ols(formula="pc_DJ_industrial ~ China_compound_tweet + China_compound_script", data=analysis_all[analysis_all['dummy19']==1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'China_compound_tweet',fig=fig)
print()

print('Regress pc_DJ_industrial on number of keyword "China" in tweets and script' +/
      'in year 2020: insignificant for both tweet and script')
result = sm.ols(formula="pc_DJ_industrial ~ China_compound_tweet + China_compound_script", data=analysis_all[analysis_all['dummy20']==1]).fit()
print(result.summary())
