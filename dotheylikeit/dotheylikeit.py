# -*- coding: utf-8 -*-
__author__ = 'Muhammad Ghafoor'
__version__ = '1.0.1'
__email__ = "mdhgafoor@outlook.com"

"""
File Name: dotheylikeit.py
Description: Natural Langauge Processing project that analyzes the sentiment of a tweet (or sentence).
This script performs sentiment analysis on tweets using a Convolutional Neural Network (CNN) model.
First a dataset of 1.6 million tweets is loaded and cleaned to remove noise like URLs, mentions, and punctuation. 
Next, the tweets are tokenized and padded to ensure unfiorm input length for the model. 
The script also supports loading a precleaned dataset. Once the data is pre-processed, the script builds and trains 
a CNN model. This process invludes loading pre-trained word embeddings from GloVe, constructing a CNN with layers for
embedding, convolution, pooling, and dense output. This model is compiled with an appropriate loss function and optimzer,
and then trained on the pre-processed training data. The trained model is saved for later use.
To test the model, the test dataset is used to determine the model's accuracy and loss. Additionally, for the user,
a random sample of tweets is selected, its sentiment predicted and compared against its actual sentiment and displayed for the user 
in a figure. There is also support to predict the sentiment of a single tweet. Overall, this script provides a comprehensive workflow 
for sentiment analysis on tweets, from data preparation to model training, evaluation, and individual tweet prediction. 
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import re 
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Dense,Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model


class DoTheyLikeIt:
    """
    A class for sentiment analysis of tweets using a CNN model.
    """

    MAX_TWEET_LENGTH = 25

    def __init__(self):
        """
        Initialize class and loads tokenizer, stemmer, lemmatizer, and essential parameters.
        """
        self.tokenizer = Tokenizer(oov_token="<UNK>")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.label_map = {0:0, 2:2, 4:1} 
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    
    def load_data_set(self, precleaned_data=False):
        """
        Loads testing and training datasets. If dataset requires cleaning, original dataset from
        Kaggle is loaded and cleaned. Otherwise, precleaned dataset pkl files and tokenizer are loaded.

        Args: 
            precleaned_data: [bool, optional] Flag for whether to load pre-cleaned data or load and clean the raw dataset
        """
        if precleaned_data:
            try:
                self.train_df = pd.read_pickle('trainingandtestdata/training.1600000.processed.noemoticon.Cleaned.pkl')
                self.test_df = pd.read_pickle('trainingandtestdata/testdata.manual.2009.06.14.Cleaned.pkl')
                self.train_df = self.train_df.sample(n=100000)
                with open('trainingandtestdata/tokenizer.pickle', 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                self.test_df = self.test_df[self.test_df['Sentiment'] != 2]
            except FileNotFoundError as e:
                print('Unable to open training and test data files. Please verify Precleaned data flag choice.')
                exit()
        else:   
            self.train_df = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='latin-1')
            self.test_df = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', encoding='latin-1')
            self.train_df.columns = ['Sentiment','ID','Date','Topic','Username','Tweet']
            self.test_df.columns = ['Sentiment','ID','Date','Topic','Username','Tweet']
            self.clean_data()
            self.train_df = self.train_df.sample(n=100000)
    

    def clean_data(self):
        """
        Cleans training and test datasets and saves as pkl files. 
        Cleaning procedure is as follows: 
            1. Remove all symbols, URLs, Hashtags, mentions etc.
            2. Pass through stemmer and lemmatizer
            3. Tokenize tweet
            4. Pad the tokenized tweet to create uniform length parameters
            5. Save tokenizer and datasets as pkl files
        """
        self.train_df['Cleaned Tweet'] = ''
        self.test_df['Cleaned Tweet'] = ''
        self.train_df['Tweet Sequence'] = ''
        self.test_df['Tweet Sequence'] = ''

        for index, row in self.test_df.iterrows():
            cleaned_tweet = row['Tweet']
            cleaned_tweet = re.sub(r"http\S+|www\S+|https\S+", '', cleaned_tweet, flags=re.MULTILINE)  # Remove URLs
            cleaned_tweet = re.sub(r'\@\w+|\#', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'[^\w\s]', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'\d+', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet)
            cleaned_tweet = nltk.word_tokenize(cleaned_tweet)
            cleaned_tweet = [word for word in cleaned_tweet if word not in self.stop_words]
            cleaned_tweet = [self.stemmer.stem(word) for word in cleaned_tweet]
            cleaned_tweet = [self.lemmatizer.lemmatize(word) for word in cleaned_tweet]
            cleaned_tweet = ' '.join(cleaned_tweet)
            self.test_df.loc[index, 'Cleaned Tweet'] = cleaned_tweet
        
        for index, row in self.train_df.iterrows():
            cleaned_tweet = row['Tweet']
            cleaned_tweet = re.sub(r"http\S+|www\S+|https\S+", '', cleaned_tweet, flags=re.MULTILINE)  # Remove URLs
            cleaned_tweet = re.sub(r'\@\w+|\#', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'[^\w\s]', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'\d+', '', cleaned_tweet)
            cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet)
            cleaned_tweet = nltk.word_tokenize(cleaned_tweet)
            cleaned_tweet = [word for word in cleaned_tweet if word not in self.stop_words]
            cleaned_tweet = [self.stemmer.stem(word) for word in cleaned_tweet]
            cleaned_tweet = [self.lemmatizer.lemmatize(word) for word in cleaned_tweet]
            cleaned_tweet = ' '.join(cleaned_tweet)
            self.train_df.loc[index, 'Cleaned Tweet'] = cleaned_tweet
        
        self.tokenizer.fit_on_texts(pd.concat([self.train_df['Cleaned Tweet'], self.test_df['Cleaned Tweet']]))
        self.train_df['Tweet Sequence'] = self.tokenizer.texts_to_sequences(self.train_df['Cleaned Tweet'])
        self.train_df['Tweet Sequence'] = pad_sequences(self.train_df['Tweet Sequence'].tolist(), maxlen=self.MAX_TWEET_LENGTH, padding='post', truncating='post').tolist()
        
        self.test_df['Tweet Sequence'] = self.tokenizer.texts_to_sequences(self.test_df['Cleaned Tweet'])
        self.test_df['Tweet Sequence'] = pad_sequences(self.test_df['Tweet Sequence'].tolist(), maxlen=self.MAX_TWEET_LENGTH, padding='post', truncating='post').tolist()
        
        self.test_df = self.test_df[self.test_df['Sentiment'] != 2]

        self.train_df['Sentiment'] = self.train_df['Sentiment'].map(self.label_map)
        self.test_df['Sentiment'] = self.test_df['Sentiment'].map(self.label_map)

        self.train_df.to_pickle('trainingandtestdata/training.1600000.processed.noemoticon.Cleaned.pkl')
        self.test_df.to_pickle('trainingandtestdata/testdata.manual.2009.06.14.Cleaned.pkl')
        
        with open('trainingandtestdata/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    def load_glove_embedding(self):
        """
        Load glove embeddings
        """
        self.embeddings_index = {}
        with open('embedding/glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        self.embedding_dim = 100
        self.embeddings_index["<UNK>"] = np.zeros((self.embedding_dim,))
    

    def create_embedding_matrix(self):
        """
        Create embedding matrix
        """
        vocab_size = len(self.tokenizer.word_index) + 1  
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        
        
    def build_model(self, vocab_size, embedding_dim=100):
        """
        Build CNN model for training

        Args:
            vocab_size: [int] Number of unique words in dataset after pre-processing
            embedding_dim: [int, optional] Size of vector used to represent each word
        """
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, weights=[self.embedding_matrix], trainable=True),
            Conv1D(132, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(2, activation='softmax', kernel_initializer='glorot_uniform') 
        ])
        self.lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])


    def train_model(self):
        """
        Execute model training
        """
        X_train = np.array(self.train_df['Tweet Sequence'].tolist())
        y_train = np.array(self.train_df['Sentiment'])
        classweight = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(classweight))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping, self.lr_scheduler], class_weight=class_weight_dict)
        self._save_model('dotheylikeit/sentimentmodel.keras')
    

    def _save_model(self, model_filedir):
        """
        Save model as a keras file

        Args:
            model_filedir: [str] directory for where model will be saved
        """
        self.model.save(model_filedir)
        

    def test_model(self, sample_size=15, model_filepath=None, save_filepath=None):
        """
        Test trained model against pre-processed testing dataset. Create a figure with 15 
        randomized tweets and their predicted sentiment alongide its actual sentiment.

        Args:
            sample_size: [int, optional] Number of samples to showcase in figure
            model_filepath: [str, optional] Directory location of model
            save_filepath: [str, optional] Directory for where to save results figure
        """
        if model_filepath:
            self.model = load_model(model_filepath)
        
        if not save_filepath:
            save_filepath = 'dotheylikeit/sentimentmodelresults.png'
        
        X_test = np.array(self.test_df['Tweet Sequence'].tolist())
        y_test = np.array(self.test_df['Sentiment'].tolist())
        for sequence in X_test:
            for i, word_index in enumerate(sequence):
                if word_index >= len(self.tokenizer.word_index):
                    sequence[i] = 0

        self.loss, self.accuracy = self.model.evaluate(X_test, y_test)
        
        results_data = []
        sample_indices = np.random.choice(self.test_df.index, sample_size, replace=False)
        sample_tweets = self.test_df.loc[sample_indices]

        for idx, row in sample_tweets.iterrows():
            tweet = np.array([row['Tweet Sequence']])
            true_sentiment = 'Positive' if row['Sentiment'] == 1 else "Negative"
            neg_prob, pos_prob = self.model.predict(tweet)[0]
            prediction = "Positive" if pos_prob > neg_prob else "Negative"
            results_data.append([row['Tweet'], true_sentiment, prediction])
        
        fig, ax = plt.subplots(figsize=(15,8))
        ax.axis('off')
        table = ax.table(cellText=results_data, colLabels=['Tweet', 'True Sentiment','Predicted Sentiment'], 
                         loc='center', cellLoc='left', colWidths=[.80,.1,.1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2,1.2)
        ax.set_title(f"Sentiment Analysis Model Results - Accuracy: {round(self.accuracy*100,2)}%")
        fig.savefig(save_filepath)


    def test_single(self, model_filepath=None, test_tweet=None):
        """
        Test a single tweet against the trained model. 
        Can be used for various purposes like testing or API calls in future.

        Args:
            model_filepath: [str, optional] Directory of where model is saved
            test_tweet: [str, optional] String to be tested in the model
        """
        if model_filepath:
            self.model = load_model(model_filepath)
    
        cleaned_tweet = test_tweet
        cleaned_tweet = re.sub(r"http\S+|www\S+|https\S+", '', cleaned_tweet, flags=re.MULTILINE)  # Remove URLs
        cleaned_tweet = re.sub(r'\@\w+|\#', '', cleaned_tweet)
        cleaned_tweet = re.sub(r'[^\w\s]', '', cleaned_tweet)
        cleaned_tweet = re.sub(r'\d+', '', cleaned_tweet)
        cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet)
        cleaned_tweet = nltk.word_tokenize(cleaned_tweet)
        cleaned_tweet = [word for word in cleaned_tweet if word not in self.stop_words]
        cleaned_tweet = [self.stemmer.stem(word) for word in cleaned_tweet]
        cleaned_tweet = [self.lemmatizer.lemmatize(word) for word in cleaned_tweet]
        cleaned_tweet = [' '.join(cleaned_tweet)]
        tokenized_tweet = self.tokenizer.texts_to_sequences(cleaned_tweet)
        padded_tweet = pad_sequences(tokenized_tweet, maxlen=self.MAX_TWEET_LENGTH, padding='post', truncating='post')

        neg_prob, pos_prob = self.model.predict(padded_tweet)[0]

        prediction = "Positive" if pos_prob > neg_prob else "Negative"
            
        return prediction


if __name__ == '__main__':
    # Example on how to run the code

    # Instantiate Class and Load dataset
    dtli = DoTheyLikeIt()
    dtli.load_data_set(precleaned_data=True)

    #ONLY IF TRAINING:
    # Load glove embedding, create embedding matrix, build model, and train
    dtli.load_glove_embedding()
    dtli.create_embedding_matrix()
    dtli.build_model(len(dtli.tokenizer.word_index) + 1)
    dtli.train_model()

    #ONLY IF TESTING
    # Test model against dataset
    dtli.test_model(15, model_filepath='dotheylikeit/sentimentmodel.keras', save_filepath='dotheylikeit/sentimentmodelresults.png')

    # Test model against single tweet
    # dtli.test_single(model_filepath='dotheylikeit/sentimentmodel.keras', test_tweet='I like you')