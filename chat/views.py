from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
# from crypt import methods
# from email.policy import default
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import json
import random
import re
import os
app = Flask(__name__)
CORS(app)

module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, 'data.json')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
intents = json.loads(open(file_path).read())

def index(request):
    return HttpResponse("Welcome to chatbot")

def bestFit(query):
    maxThreshold = 0.0
    currentRatio = 0.0
    foundIntent = {}
    # print(intents['intents'],end='/n/n')
    for intentObject in intents['intents']:
        # print(intentObject,end='/n/n')
        for phrase in intentObject['training-phrases']:
            # print(" ")
            currentRatio = SequenceMatcher(
                None, query.lower(), phrase.lower()).ratio()
            # print(query,phrase,currentRatio,end='/n/n')
            if currentRatio >= maxThreshold:
                maxThreshold = currentRatio
                foundIntent = intentObject
                # print(maxThreshold,foundIntent)
    # del foundIntent['training-phrases']
    return foundIntent


def oDataPackage(arr):
    # print((arr))
    oData = {
        'd': {
            'results': arr
        }
    }
    return oDatas
# @app.route('/returnjson/', defaults={'key':None}, methods=['GET'])
# @app.route('/returnjson/<key>', methods=['GET'])


def getJSONTemplate(key, response):
    if key == 'AdaptiveCard':
        jsonObj = {
            "text": "Adaptive card 101",
            "type": "response",
            "view": "AdaptiveCard",
            "context": "fill-details",
            "content": {
                "type": "AdaptiveCard",
                "bodyText": "This is body of the card",
                "value": [
                    {
                        "text": "Birth Date",
                        "type": "Date",
                        "value": [{
                            "start": "1990",
                            "end": "2010"
                        }]
                    }, {
                        "text": "Upload Your Resume",
                        "type": "File",
                        "value": [{
                            "type": ["pdf", "jpg"]
                        }]
                    }, {
                        "text": "Enter your Preference:",
                        "type": "CheckBox",
                        "value": [{
                            "text": "WFO"
                        }, {
                            "text": "WFH"
                        }, {
                            "text": "Hybrid"
                        }]
                    }, {
                        "text": "Ready to relocate to Ahmedabad with 12L of package?",
                        "type": "Button",
                        "value": [{
                            "text": "Yes"
                        }, {
                            "text": "No"
                        }, {
                            "text": "May be"
                        }]
                    }, {
                        "text": "Select your primary Skill",
                        "type": "List",
                        "value": [{
                            "text": "SAP Full Stack developer"
                        }, {
                            "text": "SAP Backend Developer"
                        }, {
                            "text": "SAP Frontend Developer"
                        }]
                    }]
            }
        }
    elif key == 'CheckBox':
        jsonObj = {
            "text": "Please select your suitable approach",
            "type": "response",
            "view": "Basic",
            "content": {
                "type": "CheckBox",
                    "value": [{
                        "text": "WFO"
                    }, {
                        "text": "WFH"
                    }, {
                        "text": "Hybrid"
                    }]
            }
        }
    elif key == 'Button':
        jsonObj = {
            "text": "Are you willing to relocate?",
            "type": "response",
            "view": "Basic",
            "content": {
                "type": "Button",
                "value": [{
                    "text": "Yes"
                }, {
                    "text": "No"
                }]
            }
        }
    elif key == 'List':
        jsonObj = {
            "text": "Enter your skills",
            "type": "response",
            "view": "Basic",
            "content": {
                "type": "List",
                "value": [{
                    "text": "SAP Fullstack"
                }, {
                    "text": "SAP FrontEnd"
                }, {
                    "text": "SAP BackEnd"
                }]
            }

        }
    elif key == 'Date':
        jsonObj = {
            "text": response,
            "type": "response",
            "view": "Basic",
            "content": {
                "type": "Date",
                "value": [{
                    "start": "1998",
                    "end": "2010"
                }]
            }
        }
    elif key == 'File':
        jsonObj = {
            "text": "Upload your resume",
            "type": "response",
            "view": "Basic",
            "content": {
                "type": "File",
                "value": [{
                    "type": ["pdf", "jpg"]
                }]
            }
        }
    else:
        jsonObj = {
            "intent": "",
            "context": "",
            "text": response,
            "view": "Basic",
            "type": "response",
                    "content": {
                        "type": ""
                    }
        }
    return jsonObj
# @app.route('/identifyIntent/<query>',methods=["GET"])


def IdentifyIntent(query):
    return bestFit(query)


def nltk_pos_tagging(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def remove_stopwords(words):
    return [word for word in words if not word.lower() in stop_words]
# @app.route('/nlp/<query>',methods=["GET"])


def applyNLP(query):
    # tokenization
    query = re.sub(r'[^a-zA-Z]',' ',query)
    words = nltk.word_tokenize(query)
    # stopwords removal
    words = remove_stopwords(words)
    nltk_tagged = nltk.pos_tag(words)
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagging(x[1])), nltk_tagged)
    # print(wordnet_tagged)
    nlp_words = []
    for word, tag in wordnet_tagged:
        if tag != None:
            nlp_words.append(lemmatizer.lemmatize(word.lower(), tag))
        else:
            nlp_words.append(lemmatizer.lemmatize(word.lower()))
    # print(" ".join(nlp_words))
    return " ".join(nlp_words)


def salesOrderDialog():
    arr = []
    # 1 input collection : SO number
    jsonObj = getJSONTemplate(
        key="", response="Please enter Sales Order number")
    jsonUpdate = {"intent": "sales-order"}
    jsonObj.update(jsonUpdate)
    jsonUpdate = {"context": "sales"}
    jsonObj.update(jsonUpdate)
    arr.append(jsonObj)

    # create OData type of response
    return oDataPackage(arr)


# @app.route('/processInput/<query>', methods=["GET"])
def processInput(request,query):
    # Apply NLP pipeline
    processedQuery = applyNLP(query)
    # print("# Apply NLP pipeline",processedQuery)
    # Apply ML to identify Intent
    if processedQuery:
        intentObj = IdentifyIntent(processedQuery)
    else:
        intentObj = IdentifyIntent(query)
    # print("# Apply ML to identify Intent",intentObj)
    if intentObj['intent'] == "sales-order":
        return JsonResponse(salesOrderDialog())
    elif intentObj['intent'] == "greeting":
        return JsonResponse(oDataPackage(arr=[getJSONTemplate("", random.choice(intentObj['responses']))]))
    else:
        return JsonResponse(oDataPackage(arr=[getJSONTemplate("", "I could not understand it. I am still learning...")]))


vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
# @app.route("/train_data", methods=["GET"])
def train_data(request):
    data_file = open('data.json').read()
    intents = json.loads(data_file)
    data = []
    data_temp = []
    label_int_map = []
    for intent in intents['intents']:
        if intent['intent'] not in label_int_map:
            label_int_map.append(intent['intent']) 

    for intent in intents['intents']:
        label = intent['intent']
        phrases = []
        for phrase in intent['training-phrases']:
            processed_phrase = applyNLP(phrase)
            if not processed_phrase:
                processed_phrase = phrase
            data.append([processed_phrase, label_int_map.index(label)])

    df = pd.DataFrame(data, columns=['Phrases', 'label'])
    X = df.iloc[:, 0].values
    
    y = df.iloc[:, 1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print("X_train",X_train, "X_test",X_test)
    classifier = MultinomialNB()
    classifier.fit(pd.get_dummies(X_train), y_train)

    # print(X_test)
    y_pred = classifier.predict(pd.get_dummies(X_test))
    # classfication_report = classfication_report(y_test, y_pred)
    # accuracy_score = accuracy_score(y_test, y_pred)
    # print(classfication_report, accuracy_score)
    print(y_pred)
    # t(df)
    # tfidf_vectors = vectorizer.fit_transform(data_temp)
    # cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)
    # similar_response_index = cosine_similarities.argsort()[0][-2]
    # print(tfidf_vectors,similar_response_index)


if __name__ == '__main__':
    train_data()
    # app.run(debug=True)
