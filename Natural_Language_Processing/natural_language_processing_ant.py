# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
"""
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#'/t' shows columns sperated by a tab
# quoting = 3 - ignore double quotes

"""
# Cleaning the Texts
"""
import re #library to clean text
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) #want to keep all cases of A-Z | From where
review = review.lower()  #put all letters to lower case

import nltk #natural language processing Library
nltk.download('stopwords') #download stop words package
from nltk.corpus import stopwords

#create a for loop to go through every word in the review and make sure there's no stopwords.

review = review.split() #split the review (which is a string) into different words

review = [word for word in review if not word in set(stopwords.words('english'))] 
#for each word in the review array, make sure they're not in the stopwwords list of english words



"""
# Creating the Bag of Words model
"""



"""
# Splitting the dataset into the Training set and Test set
"""



"""
# Fitting Naive Bayes to the Training set
"""



"""
# Predicting the Test set results
"""




"""
# Making the Confusion Matrix
"""