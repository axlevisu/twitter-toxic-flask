from flask import Flask, jsonify, request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.manifold import TSNE
from keras.models import load_model


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
# print df.head()
loaded_model = load_model('myy_model.h5')
loaded_model._make_predict_function()
df['comment_text'] = df['comment_text'].map(lambda x: clean_text(x))
labels = df[CLASSES]
### Create sequence
vocabulary_size = 5000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['comment_text'])
app = Flask(__name__)

# return 'Advice for ',referenceId
@app.route('/analysis/', methods=['GET', 'POST'])
def analysis():
    # response = {"Payload":{"referenceId":referenceId, "strategyId":randint(0, 5)}}
    # print request.args
    tweet = request.args.get('tweet',0)
    print tweet
    tweet = clean_text(str(tweet))
    sequences = tokenizer.texts_to_sequences([tweet])
    data = pad_sequences(sequences, maxlen=50)
    print data
    prediction = loaded_model.predict(data)[0]
    send_tweet = "On a scale of 1 to 10."
    for i in xrange(0,6):
    	if i == 4:
    		send_tweet = send_tweet + " " + str(round(10*prediction[i],1)) + " " + CLASSES[i] + " and"
    	elif i ==5:
    		send_tweet = send_tweet + " " + str(round(10*prediction[i],1)) + " " + CLASSES[i] + "."
    	else:
    		send_tweet = send_tweet + " " + str(round(10*prediction[i],1)) + " " + CLASSES[i] + ","
    return jsonify({"report": send_tweet})
    # return "henlo"

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=7000,debug=True)