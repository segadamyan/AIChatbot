import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')


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
    print(result)
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I couldn't understand sorry, please say another way"
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
    print(ints)
    res = get_response(ints, intents)
    print(res)
