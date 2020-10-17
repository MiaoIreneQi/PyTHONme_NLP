# Group: PyTHONme_NLP
# Project topic: presidential portfolio
# Authors in collaboration: QI Miao Irene, CHEN Jingshu David, JIANG Binghan, BAO Enqi Bruce

# Temporary message: Hey guys, this is a python file. Please type your codes below.

import reuqests
import pandas as pd
import numpy as np
import nltk
# We first scrape the transcripts of Trump's past speeches.
link_1_Trump = 'https://www.rev.com/blog/transcripts/donald-trump-campaign-rally-greenville-nc-transcript-october-15'
r = requests.get(link_1_Trump)
# Obviously, we will end up with something very messy from the step above, so we do some sorting.
