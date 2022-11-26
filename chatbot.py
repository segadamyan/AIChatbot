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
from datetime import datetime


class Languages(Enum):
    Python = ("python", "py")
    JAVA = ("java", "java")
    CPP = ("cpp", "cpp")


def SETUP_NLTK():
    nltk.download("punkt")
    nltk.download("amw1-s4")
    nltk.download("wordnet")


class Chatbot:
    SORTING_ALGORITHMS = {
        "insertion_sort": (
            "Insertion sort is a simple sorting algorithm that works similar to the way you sort playing cards in "
            "your hands. The array is virtually split into a sorted and an unsorted part. Values from the unsorted "
            "part are picked and placed at the correct position in the sorted part.",
            "O(N^2)"),
        "bubble_sort": (
            "Bubble Sort is the simplest sorting algorithm that works by repeatedly swapping the adjacent elements if "
            "they are in the wrong order. This algorithm is not suitable for large data sets as its average and "
            "worst-case time complexity is quite high.",
            "O(N2)"),
        "selection_sort": (
            "The selection sort algorithm sorts an array by repeatedly finding the minimum element (considering "
            "ascending order) from the unsorted part and putting it at the beginning. ",
            "O(N2)"),
        "merge_sort": (
            "The Merge Sort algorithm is a sorting algorithm that is based on the Divide and Conquer paradigm. In "
            "this algorithm, the array is initially divided into two equal halves and then they are combined in a "
            "sorted manner.",
            "O(N log(N))"),
        "quick_sort": (
            "Like Merge Sort, QuickSort is a Divide and Conquer algorithm. It picks an element as a pivot and "
            "partitions the given array around the picked pivot. There are many different versions of quickSort that "
            "pick pivot in different ways.",
            "O(N*logN)")
    }

    def __init__(self, intent: str):
        self.lemmatizer = WordNetLemmatizer()
        self.intent = json.loads(open(f'{intent}.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot.h5')
        self.previousExtension = ""

    def __clean_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def __bag_of_word(self, sentence):
        sentence_words = self.__clean_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def __predict(self, sentence):
        bow = self.__bag_of_word(sentence)
        if np.all((bow == 0)):
            return []
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.2
        result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def __get_response(self, intents_list, intents_json):
        if len(intents_list) == 0:
            return "Can you try asking it in a different way?"
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        result = None
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def predict(self, message):
        intents_list = self.__predict(message)
        response = self.__get_response(intents_list, self.intent)
        if response.startswith("[func]"):
            function = response.split(".")[1]
            if function == "time":
                response = datetime.now()

        elif response.startswith("[code]"):
            response = response[6:]
            language, filename = response.split(".")
            if language == Languages.Python.value[0]:
                extension = Languages.Python.value[1]
            elif language == Languages.JAVA.value[0]:
                extension = Languages.JAVA.value[1]
            elif language == Languages.CPP.value[0]:
                extension = Languages.CPP.value[1]
            else:
                extension = "txt"
            if not self.previousExtension == "":
                os.remove(f"response.{self.previousExtension}")
            shutil.copyfile(f'resources/code/{language}/{filename}.{extension}', f'response.{extension}')
            self.previousExtension = extension
            response = Chatbot.SORTING_ALGORITHMS[filename][0] + ' \nComplexity: ' + \
                       Chatbot.SORTING_ALGORITHMS[filename][1]
        elif response.startswith("Cancelled"):
            if not self.previousExtension == "":
                os.remove(f"response.{self.previousExtension}")
                self.previousExtension = ""
        return response

    def start(self):
        while True:
            message = input("")
            intents_list = self.__predict(message)
            response = self.__get_response(intents_list, self.intent)
            if response.startswith("[func]"):
                function = response.split(".")[1]
                if function == "time":
                    response = datetime.now()

            elif response.startswith("[code]"):
                response = response[6:]
                language, filename = response.split(".")
                if language == Languages.Python.value[0]:
                    extension = Languages.Python.value[1]
                elif language == Languages.JAVA.value[0]:
                    extension = Languages.JAVA.value[1]
                elif language == Languages.CPP.value[0]:
                    extension = Languages.CPP.value[1]
                else:
                    extension = "txt"
                if not self.previousExtension == "":
                    os.remove(f"response.{self.previousExtension}")
                shutil.copyfile(f'resources/code/{language}/{filename}.{extension}', f'response.{extension}')
                self.previousExtension = extension
                response = Chatbot.SORTING_ALGORITHMS[filename][0] + ' \nComplexity: ' + \
                           Chatbot.SORTING_ALGORITHMS[filename][1]
            elif response.startswith("Cancelled"):
                if not self.previousExtension == "":
                    os.remove(f"response.{self.previousExtension}")
                    self.previousExtension = ""
            print(response)


if __name__ == "__main__":
    chatbot = Chatbot("intents")
    chatbot.start()
