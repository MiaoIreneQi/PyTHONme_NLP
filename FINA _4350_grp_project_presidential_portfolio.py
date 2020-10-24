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

# We first scrape the transcripts of Trump's past speeches.
transcript = 'https://www.rev.com/blog/transcripts/{}'
final_debate_raw = requests.get(transcript.format('donald-trump-joe-biden-final-presidential-debate-transcript-2020')).text

# Obviously, we will end up with something very messy from the step above, so we do some cleaning.
final_debate_s = BeautifulSoup(final_debate_raw, 'lxml')
cleaned_text_final_debate = [tag.text for tag in final_debate_s.find_all('p')]

#merge text into one single file.
single = ''.join(cleaned_text_final_debate)

#save the transcript to file trumscript, using pickle.
import pickle

transcript1 = regexp_tokenize(single, r'(\w+)')
with open('transcript1.pickle','wb') as trumpscript:
    pickle.dump(transcript1,trumpscript,pickle.HIGHEST_PROTOCOL)

with open('transcript1.pickle','rb') as trumpscript:
    transcript1=pickle.load(trumpscript)

    
    
# Obtain the title of each page for page 1    
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk    

r = \
    requests.get(
        'https://www.rev.com/blog/transcript-category/donald-trump-transcripts?view=all', timeout=5)
    
clean_transcript_p1 = BeautifulSoup(r.text, 'lxml')

tag_name_list = {tag.name for tag in clean_transcript_p1.find_all(True)}

title_p1 = {tag.text for tag in clean_transcript_p1.find_all(['strong'])}
title_p1.remove("Help Us Improve the Rev Transcript Library!")




# loop for differnt pages

address = "https://www.rev.com/blog/transcript-category/donald-trump-transcripts/page/{}?view=all"

web_list = []
for i in range (2,34):
    web_list.append(address.format(i))
    


# Combination (in progress)
for link in web_list:
        full_web_pages = requests.get(link, timeout=5)
        clean_full_web_page = BeautifulSoup(full_web_pages, 'lxml')
        title_p = {tag.text for tag in clean_full_web_page.find_all(['strong'])}
        title_p.remove("Help Us Improve the Rev Transcript Library!")
        
        for item in title_p:
            address.requests(item, timeout=5)
       
#creat a list to contain all the titles from all the pages (loop) : 
title_list = []
for web in web_list:
    title_list.append([tag.text for tag in BeautifulSoup(requests.get(web).text).find_all('strong')])

#remove the unnecessary titles from the list (loop): 
for sublist in title_list:
    sublist.remove(''Help Us Improve the Rev Transcript Library!')

#from list in list to one list.
title_list_unnested = [item for sublist in title_list for item in sublist]

import string
ex_ti.lower().translate(str.maketrans('','',string.punctuation))
