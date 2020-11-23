# Group: PyTHONme_NLP
# Project topic: presidential portfolio construction
# Authors in collaboration: QI Miao Irene, CHEN Jingshu David, JIANG Binghan Stephanie, BAO Enqi Bruce

# Temporary message: Hey guys, this is a python file. Please type your codes below.

#Object1: main branch - web scrapping from webpage: https://www.rev.com/blog/transcript-category/donald-trump-transcripts

#to start with, import packages needed. 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from numpy import nan
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
                   

#### The followings are the start of our FORMAL coding lol:

# Step 1: obtain the titles in Page 1
#import requests
#from bs4 import BeautifulSoup
#import pandas as pd
#import numpy as np
#import nltk
#from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
                   
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

wnl = WordNetLemmatizer()

lemmatized_counting = [Counter(article) for article in lemmatized]

ngs = [ngrams(element, 2) for element in no_stops_collection]
gram_2_list = [[' '.join(ng) for ng in element] for element in ngs]
counting_gram_2 = [Counter(article) for article in gram_2_list]


#Analyze (mannually) selected keywords
def word_count(word):
    counting = []
    for element in lemmatized_counting:
        if word in element:
            counting.append(element[word])
        else:
            counting.append(0)
    return counting

keywords = ['China', 'tariff', 'Xi', 'Putin', 'tax', 
            'COVID', 'virus', 'fake', 'abortion', 'Russia']

keyword_dic = {}
for keyword in keywords:
    keyword_dic[keyword] = word_count(keyword)

keyword_df = pd.DataFrame([keyword_dic[keyword] for keyword in keyword_dic])
keyword_df = keyword_df.T
keyword_df.rename(columns = dict(zip(range(len(keywords)),keywords)), inplace = True)

keyword_df.insert(loc = 0 , column = 'Date', 
                  value = table_for_all_articles.Date.tolist())

keyword_df.Date = pd.to_datetime(keyword_df.Date)

def date_combine_keyword(word):
    tempo = keyword_df.groupby('Date')[word].apply(list)
    date_distinct = tempo.tolist()
    date_distinct_sum = []

    for element in date_distinct:
        date_distinct_sum.append(sum(element))
    return date_distinct_sum

keyword_df2 = pd.DataFrame({keyword : date_combine_keyword(keyword) 
                            for keyword in keywords})

keyword_df2.insert(loc = 0, column = 'Date', 
                   value = sorted(list(set(keyword_df.Date))))


#code from David&Bruce on Nov 22
data_new = pd.read_excel('data_new.xlsx')
data_new.replace([0], nan, inplace = True)
keyword_df2.set_index('Date', inplace = True)
t_index = pd.date_range('2017-01-03','2020-11-13')
keyword_df3 = keyword_df2.reindex(t_index, fill_value = nan)

keyword_df3.reset_index(inplace = True)
keyword_df3.rename(columns = {'index' : 'Date'}, inplace = True)


# preparing Alibaba's data for merging
alibaba = pd.read_excel('alibaba.xlsx')
alibaba.Date = pd.to_datetime(alibaba.Date)
alibaba.set_index('Date', inplace = True)
alibaba2 = alibaba.reindex(t_index, fill_value = nan)
alibaba2.reset_index(inplace = True)
alibaba2.rename(columns = {'index' : 'Date'}, inplace = True)
analysis_all = pd.merge(analysis_all, alibaba2, on = 'Date')

# preparing data_new for merging
data_new = pd.read_excel('data_new.xlsx')
data_new.rename(columns = {'索引' : 'index', 'Dow_Jones_工业' : 'DJ_industrial', '美国:道琼斯公用事业平均指数' : 'DJ_public', '美国:威尔希尔美国房地产投资信托市场总指数': 'WS_housing', '美国:能源产业ETF波动率指数' : 'Energy_ETFVIX'}, inplace = True)
data_new.Date = pd.to_datetime(data_new.Date)
data_new.set_index('Date', inplace = True)
data_new2 = data_new.reindex(t_index, fill_value = nan)
data_new2.reset_index(inplace = True)
data_new2.rename(columns = {'index' : 'Date'}, inplace = True)
analysis_all = pd.merge(analysis_all, data_new2, on = 'Date')

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




import statsmodels.formula.api as sm
import statsmodels.api as sm1
# run DJ_industry on compund_tweet if year ==2017: 
#we found significantly positive result! Although the magnitude is very small
result = sm.ols(formula="pc_DJ_industrial_lead ~ compound_tweet", data=analysis_all[analysis_all['dummy17'] == 1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'compound_tweet',fig=fig)

# run DJ_industry on compund_tweet if year ==2018: 
#we found significantly negative result! Although the magnitude is very small
result = sm.ols(formula="pc_DJ_industrial_lead ~ compound_tweet", data=analysis_all[analysis_all['dummy18'] == 1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'compound_tweet',fig=fig)

# run DJ_industry on compund_tweet if year ==2019: 
#we found insignificant result
result = sm.ols(formula="pc_DJ_industrial_lead ~ compound_tweet", data=analysis_all[analysis_all['dummy19'] == 1]).fit()
print(result.summary())

# run DJ_industry on compund_tweet if year ==2020: 
#we found insignificant result
result = sm.ols(formula="pc_DJ_industrial_lead ~ compound_tweet", data=analysis_all[analysis_all['dummy20'] == 1]).fit()
print(result.summary())

keywords = ['China', 'tariff', 'Xi', 'Putin', 'tax', 
            'COVID', 'virus', 'fake', 'abortion', 'Russia']

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

#Merge dataframes of keywords from both script and tweets
keyword_analysis = pd.merge(keyword_df3, keyword_tweet_df2, on = 'Date', 
                            suffixes = ('_script', '_tweet'))

analysis_all = pd.merge(analysis_all, keyword_analysis, on = 'Date')

analysis_all['China_compound_script'] = analysis_all['China_script'] +\
    analysis_all['Xi_script']

analysis_all['China_compound_tweet'] = analysis_all['China_tweet'] +\
    analysis_all['Xi_tweet']

analysis_all['Russia_compound_script'] = analysis_all['Russia_script'] +\
    analysis_all['Putin_script']

analysis_all['Russia_compound_tweet'] = analysis_all['Russia_tweet'] +\
    analysis_all['Putin_tweet']
    
# Regress DJ_industry_lead on number of keyword "China" in tweets and script: 
 #significantly negative for tweet, insignificant for script
result = sm.ols(formula="pc_DJ_industrial_lead ~ China_compound_tweet + China_compound_script", data=analysis_all).fit()

# Regress DJ_industry_lead on number of keyword "China" in tweets and script 
#in year 2019: significantly negative for tweet, significantly positive for 
#script (smaller magnitude than tweet)
result = sm.ols(formula="pc_DJ_industrial_lead ~ China_compound_tweet + China_compound_script", data=analysis_all[analysis_all['dummy19']==1]).fit()
print(result.summary())
fig = plt.figure(figsize=(12,8))
plots = sm1.graphics.plot_regress_exog(result, 'China_compound_tweet',fig=fig)

# Regress DJ_industry_lead on number of keyword "China" in tweets and script 
#in year 2020: insignificant for both tweet and script
result = sm.ols(formula="pc_DJ_industrial_lead ~ China_compound_tweet + China_compound_script", data=analysis_all[analysis_all['dummy20']==1]).fit()


        
