# -*- coding: utf-8 -*-
"""MovieModel.py
https://towardsdatascience.com/how-to-build-and-deploy-an-nlp-model-with-fastapi-part-1-9c1c7030d40

https://github.com/Davisy/Deploy-NLP-Model-with-FastAPI
"""

# import important modules
import os
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 

from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re #regular expression

# Download dependency
for dependency in ('stopwords',"brown","names","wordnet","averaged_perceptron_tagger","universal_tagset"):
  nltk.download(dependency)
import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)
import configparser
config_object= configparser.ConfigParser()
config_object.read("configVariable.ini")
# load data
data = pd.read_excel(config_object['Path']['input_path'])
# show top five rows of data
data.head()

"""Our dataset has 3 columns.

Id — This is the id of the review

Sentiment — either positive(1) or negative(0)

"""
"""Stemming is a technique used to extract the base form of the words by removing affixes from them. It is just like cutting down the branches of a tree to its stems. For example, the stem of the words eating, eats, eaten is eat"""

stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
      sentence = text.lower()
      tokenizer = RegexpTokenizer(r'\w+')
      tokens = tokenizer.tokenize(sentence)
      text = [w for w in tokens if not w in stop_words]
      text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)
#clean the review
print('Text-Clean-Happening')
data["cleaned_review"] = data["review"].apply(text_cleaning)

"""But before training the model, we need to transform our cleaned reviews into numerical values so that the model can understand the data. In this case, we will use the TfidfVectorizer method from scikit-learn. TfidfVectorizer will help us to convert a collection of text documents to a matrix of TF-IDF features."""

#split features and target from  data 
X = data["cleaned_review"]
y = data.sentiment.values
print('Train-Test-Split-Happening')
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)
# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[('pre_processing', TfidfVectorizer(lowercase=False)),
                ('naive_bayes', MultinomialNB())])
# train the sentiment classifier 
sentiment_classifier.fit(X_train,y_train)
# test model performance on valid data 
y_preds = sentiment_classifier.predict(X_valid)
print(accuracy_score(y_valid,y_preds))
print(classification_report(y_valid, y_preds))

output_path=config_object['Path']['output_model_path']
isExist = os.path.exists(output_path)
if not isExist:
  # Create a new directory because it does not exist
  os.makedirs(output_path)
  print("The new directory is created!")
#save model 
import joblib
print('Saving-Model....')
joblib.dump(sentiment_classifier,output_path)


