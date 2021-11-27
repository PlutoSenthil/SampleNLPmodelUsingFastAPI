# -*- coding: utf-8 -*-
"""movie_main.py
"""
# text preprocessing modules
from string import punctuation

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression

import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI
import requests as r
"""# Initializing a FastAPI App Instance

we have customized the configuration of our FastAPI application by including:

Title of the API

Description of the API.

The version of the API.
"""

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

"""# Load the NLP model


"""
import configparser
config_object= configparser.ConfigParser()
config_object.read("configVariable.ini")
# load the sentiment model
with open(config_object['Path']['output_model_path'], "rb") as f:
    model = joblib.load(f)

"""# Define a Function to Clean the Data"""

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text

"""# Create Prediction Endpoint"""

@app.get("/predict-review")
async def predict_sentiment(review: str):
    # clean the review
    cleaned_review = text_cleaning(review)
     # perform prediction
    prediction = model.predict([cleaned_review])
    probas = model.predict_proba([cleaned_review])
    output = int(prediction[0])
    print(output)
    output_probability = "{:.2f}".format(float(probas[:, output]))
    sentiments = {0: "Negative", 1: "Positive"}
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

"""FastAPI is built upon two major python libraries — Starlette(for web handling) and Pydantic(for data handling & validation). FastAPI is very fast compared to Flask because it brings asynchronous function handlers to the table.

https://tiangolo.medium.com/introducing-fastapi-fdc1206d453f

You will also need an ASGI server for production such as uvicorn
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
#     # review = "This movie was exactly what I wanted in a Godzilla vs Kong movie. It's big loud, brash and dumb, in the best ways possible. It also has a heart in a the form of Jia (Kaylee Hottle) and a superbly expressionful Kong. The scenes of him in the hollow world are especially impactful and beautifully shot/animated. Kong really is the emotional core of the film (with Godzilla more of an indifferent force of nature), and is done so well he may even convert a few members of Team Godzilla."
#     # keys = {"review": review}
#     # prediction = r.get("http://127.0.0.1:8000/predict-review/", params=keys)
#     # results = prediction.json()
#     # print(results["prediction"])
#     # print(results["Probability"])