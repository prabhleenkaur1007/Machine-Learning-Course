# -*- coding: utf-8 -*-
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

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

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

"""###############################################
# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
    
fpr, tpr, thresholds = roc_curve(y_test, y_score)"""

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1000):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




