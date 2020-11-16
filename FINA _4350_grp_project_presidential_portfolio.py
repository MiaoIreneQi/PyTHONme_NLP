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
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
                   
#Using Twitter API to find out a key word
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream

access_token = "1318443538372153345-5nKY6whmlIt3ncDpHbWZOGsGUWasL7"
access_token_secret = "9oI1XCJ9dInSp7s6HCadtgPq01Wi7XaxIsnEvJjjJObTS"
consumer_key = "5eirheaMH6P8e50RILo0uFMxl"
consumer_secret = "C6eMpU1Ibr1n7pM0kpasR4Tf8al1LboxXqcKlFM7T9EvhvkTvA"

class StuOutListener(StreamListener):
    
    def on_data(self, data):
        print(data)
        return True
    
    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    
    listener = StuOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    stream = Stream(auth, listener)
    
    stream.filter(track = ['Donald Trump','Joe Biden'])




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

counting = [Counter(article) for article in tokenize_list]

no_numeral = [[t for t in article if not t.isnumeric()] 
              for article in no_stops_collection]

wnl = WordNetLemmatizer()

lemmatized_counting = [Counter(article) for article in lemmatized]

ngs = [ngrams(element, 2) for element in no_stops_collection]
gram_2_list = [[' '.join(ng) for ng in element] for element in ngs]
counting_gram_2 = [Counter(article) for article in gram_2_list]
