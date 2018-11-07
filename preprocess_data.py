# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.manifold import TSNE
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold


INDATA_LOCATION = 'data/train.csv'

# utility definitions for easier handling of the dataset column names
TEXT_COLUMN = 'comment_text'
CLASS_TOXIC, CLASS_SEVER_TOXIC, CLASS_OBSCENE, CLASS_THREAT, CLASS_INSULT, \
    CLASS_IDENTITY_HATE = ["toxic", "severe_toxic", "obscene", "threat", \
                           "insult", "identity_hate"]
CLASSES = [CLASS_TOXIC, CLASS_SEVER_TOXIC, CLASS_OBSCENE, CLASS_THREAT, CLASS_INSULT, CLASS_IDENTITY_HATE]

# read the comments and associated classification data 
df = pd.read_csv(INDATA_LOCATION,names = ["id",TEXT_COLUMN] + CLASSES, skiprows=1)
df = df.drop('id',axis=1)
df= df.dropna()
print df.head()

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(None, string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

df['comment_text'] = df['comment_text'].map(lambda x: clean_text(x))
labels = np.array(df[CLASSES])
### Create sequence
vocabulary_size = 5000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['comment_text'])

sequences = tokenizer.texts_to_sequences(df['comment_text'])
data = pad_sequences(sequences, maxlen=50)
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# kfold = StratifiedKFold(n_splits=1, test_size=0.1, random_state=0)
# cvscores = []
# for train, test in kfold.split(data, labels.sum(axis=1)):
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, labels, validation_split=0.4, epochs=1)
model.save('myy_model.h5')
# scores = model.evaluate(data[test], labels[test], verbose=0)
# print scores
# cvscores.append(scores[1] * 100)

# print cvscores
# model.save('myy_model.h5')  # creates a HDF5 file 'my_model.h5'
# #Predict

# word = "nigger they are nigger they win"
# vocabulary_size = 5000
# tokenizer = Tokenizer(num_words= vocabulary_size)
# tokenizer.fit_on_texts(df['comment_text'])
# sequences = tokenizer.texts_to_sequences([word])
# data = pad_sequences(sequences, maxlen=50)
# print data


# loaded_model = load_model('myy_model.h5')
# loaded_model.predict(data)