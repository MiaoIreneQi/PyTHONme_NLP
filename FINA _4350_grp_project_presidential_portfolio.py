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
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
import pickle

# We first scrape the transcripts of Trump's past speeches.
#transcript = 'https://www.rev.com/blog/transcripts/{}'
#final_debate_raw = requests.get(transcript.format('donald-trump-joe-biden-final-presidential-debate-transcript-2020')).text

# Obviously, we will end up with something very messy from the step above, so we do some cleaning.
#final_debate_s = BeautifulSoup(final_debate_raw, 'lxml')
#cleaned_text_final_debate = [tag.text for tag in final_debate_s.find_all('p')]

#if we come across an empty set, that is an exception with different address. 

#merge text into one single file.
#single = ' '.join(cleaned_text_final_debate)

#save the transcript to file trumscript, using pickle.
#import pickle

#transcript1 = regexp_tokenize(single, r'(\w+)')
#with open('transcript1.pickle','wb') as trumpscript:
    #pickle.dump(transcript1,trumpscript,pickle.HIGHEST_PROTOCOL)

#with open('transcript1.pickle','rb') as trumpscript:
    #transcript1=pickle.load(trumpscript)

    
    
# Obtain the title of each page for page 1    
#import requests
#from bs4 import BeautifulSoup
#import pandas as pd
#import numpy as np
#import nltk    





     

#remove all the punctuations from all the titles, and change then into lowercase letters. Then substitute all the spaces with '-'.
#import string
#re.sub(r' ', r'-', ex_ti.lower().translate(str.maketrans('','',string.punctuation)))
                   
#Using Twitter API to find out a key word
import tweepy
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



# Step 2.1 Creating https for turining the pages (2-33)
address = "https://www.rev.com/blog/transcript-category/donald-trump-transcripts/page/{}?view=all"

last_page_candidates = []
for tag in clean_transcript_p1.find_all('a'):
    if type(tag.get('class')) == list:
        if 'page-numbers' in tag.get('class'):
            last_page_candidates.append(tag.text)

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



# Step 3 Parsing the titles
#import re
#import string             
   # remove punctuations and change upper case to lower case
#title_list_unnested_no_punct = []
#for item in title_list_unnested: 
    #title_list_unnested_no_punct.append\
        #(item.lower().translate(str.maketrans('','',string.punctuation)))
   # remove quotation mark     
#title_list_unnested_no_punct_no_quote = []
#for name in title_list_unnested_no_punct:
    #title_list_unnested_no_punct_no_quote.append\
        #(''.join(item for item in name if item not in (r"’")))
   # substitute blank spaces with hyphen.
#title_list_final = []
#for name in title_list_unnested_no_punct_no_quote:
    #title_list_final.append(re.sub(r' ',r'-',name))

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
        cleaned_article_in_paragraph.remove(' Try Rev and save time transcribing, captioning, and subtitling.')
    cleaned_article_in_paragraph.pop(-1)
    articles_in_paragraph.append(cleaned_article_in_paragraph)
    article = '\n'.join(cleaned_article_in_paragraph)
    articles.append(article)

date = [sublist.pop(0) for sublist in articles_in_paragraph]

table_for_all_articles = pd.DataFrame({'Title': title_list_unnested, 'Date': date, 'Article in paragraphs': articles_in_paragraph, 'Article continuous': articles})
#Set Title column as the index column
table_for_all_articles = table_for_all_articles.set_index('Title')
