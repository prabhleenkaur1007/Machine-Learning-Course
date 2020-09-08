# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3 to ignore double qoutes in reviews

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
# contains list of irrelevant words that we want to remove from our reviews as 
# they are not helping machine to predict whether the review is +ve or not.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # corpus means collection of text of same type
for i in range(0, 1000):# STEP-6 do the cleaning and joinig for all the 1000 reviews
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])# STEP-1 only keep the letters in the review, remove punction and numbers
    review = review.lower() #STEP-2 convert all letters to lowercase
    review = review.split() # STEP-3 a) splitting review in different words to change it from string to list of words
    ps = PorterStemmer() # STEP-4 stemmimg -taking root of the words(loved -> love)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]# STEP-3 b) include words in the review that are not in stopwords list
    review = ' '.join(review) # STEP-5 joining different words (with a space) of review list 
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# max_feature parameter to filter non-relevant words. Without this attribute we got a total of 1565 columns, now it filter the ones with the least frequency
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
# now the machine will understad the correlation of presence of some words in the review and the outcome (+ve or -ve)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)