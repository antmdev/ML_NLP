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
import nltk #natural language processing Library
nltk.download('stopwords') #download stop words package
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #want to keep all cases of A-Z | From where
    review = review.lower()  #put all letters to lower case
    review = review.split() #split the review (which is a string) into different words
    ps = PorterStemmer() #Stemmer - abbreviates words to present tense etc. like "loved" becomes "love"
    #create a for loop to go through every word in the review and make sure there's no stopwords.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    #for each word in the review array, make sure they're not in the stopwwords list of english words
    #set word makes the search quicker
    review = ' ' . join(review) #joining the words back together from list to string (append with empty space)
    corpus.append(review)


"""
# Creating the Bag of Words model
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max_features keeps only the most frequent words - removes anomolies like "Chris"
X = cv.fit_transform(corpus).toarray()  #tokenize the outputs into a hug matrix

y = dataset.iloc[:, 1].values #grab 1 or 0 good/ bads from dataset these are the dependent variables

"""
The tokenization creates a matrix of 1000 x 1500 (1500 independent variables) so each row here corresponds to one specific review. For each of those reviews we get a zero if the word doesn't appear in the review and a 1 if the word appears in the review. That gives us a classification model. The machine learning model we will train will try to understand the correlations between the presence of the words in the reviews and the outcome; if it's zero if it's a negative review, if it's a 1 then its a positiive review.
"""

"""
# Splitting the dataset into the Training set and Test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# =============================================================================
# """
# # Feature scaling - NOT REQUIRED as similar range 0,1,
# """
# from sklearn.naive_bayes import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# =============================================================================

"""
# Fitting Naive Bayes to the Training set
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""
# Predicting the Test set results
"""
y_pred = classifier.predict(X_test)

"""
# Making the Confusion Matrix
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)