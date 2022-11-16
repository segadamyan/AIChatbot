import json
import os
import pickle
import random
import shutil
from enum import Enum

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')
previousExtension = ""


class Languages(Enum):
    Python = ("python", "py")
    JAVA = ("java", "java")
    CPP = ("cpp", "cpp")


def deleteFile(filename):
    os.remove(filename)


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_word(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(bag)
    return np.array(bag)


def predict(sentence):
    bow = bag_of_word(sentence)
    if np.all((bow == 0)):
        return []
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.2
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Can you try asking it in a different way?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


while True:
    message = input("")
    ints = predict(message)
    res = get_response(ints, intents)
    if res.startswith("[code]"):

        res = res[6:]
        language, filename = res.split(".")
        extension = ""
        if language == Languages.Python.value[0]:
            extension = Languages.Python.value[1]
        elif language == Languages.JAVA.value[0]:
            extension = Languages.JAVA.value[1]
        elif language == Languages.CPP.value[0]:
            extension = Languages.CPP.value[1]
        else:
            extension = "txt"
        if not previousExtension == "":
            deleteFile(f"response.{previousExtension}")
        shutil.copyfile(f'resources/code/{language}/{filename}.{extension}', f'response.{extension}')
        previousExtension = extension
    elif res.startswith("Cancelled"):
        if not previousExtension == "":
            deleteFile(f"response.{previousExtension}")
            previousExtension = ""
    print(res)
