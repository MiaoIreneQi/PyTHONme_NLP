# Group: PyTHONme_NLP
# Project topic: presidential portfolio
# Authors in collaboration: QI Miao Irene, CHEN Jingshu David, JIANG Binghan, BAO Enqi Bruce

# Temporary message: Hey guys, this is a python file. Please type your codes below.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk
# We first scrape the transcripts of Trump's past speeches.
transcript = 'https://www.rev.com/blog/transcripts/{}'
final_debate_raw = requests.get(transcript.format('donald-trump-joe-biden-final-presidential-debate-transcript-2020')).text
# Obviously, we will end up with something very messy from the step above, so we do some cleaning.
final_debate_s = BeautifulSoup(final_debate_raw, 'lxml')
cleaned_text_final_debate = [tag.text for tag in final_debate_s.find_all('p')]
