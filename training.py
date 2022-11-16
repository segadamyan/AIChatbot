import json
import pickle
import random

import nltk
import numpy as np
from keras import regularizers
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential


class ChatbotModel:
    INTENTS = 'intents.json'
    IGNORE_LETTERS = ['?', '!', '.', ',']
    LEMMITIZER = WordNetLemmatizer()

    def __init__(self):
        self.intents = json.loads(open(ChatbotModel.INTENTS).read())
        self.words = []
        self.classes = []
        self.documents = []

    def train(self):
        self.__prepare_sentance()
        train_x, train_y = self.__prepare_for_nn()
        self.__set_up_model(train_x, train_y)

    def __prepare_sentance(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern.lower())
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [ChatbotModel.LEMMITIZER.lemmatize(word) for word in self.words if
                      word not in ChatbotModel.IGNORE_LETTERS]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        print(self.words)
        print(self.classes)

        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def __prepare_for_nn(self):
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [ChatbotModel.LEMMITIZER.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        print(train_x)
        print(train_y)
        return train_x, train_y

    def __set_up_model(self, train_x, train_y):
        model = Sequential()
        model.add(
            Dense(128, input_shape=(len(train_x[0]),), kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'), )
        model.add(Dropout(0.3))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=400, batch_size=5, validation_split=0.1,
                         verbose=1)
        model.save('chatbot.h5', hist)


if __name__ == "__main__":
    chatbot = ChatbotModel()
    chatbot.train()
    print("Done")
