# DoTheyLikeIt

Project Overview: Natural Langauge Processing project that analyzes the sentiment of a tweet (or sentence).
This script performs sentiment analysis on tweets using a Convolutional Neural Network (CNN) model via the following procedure:
1. A dataset of 1.6 million tweets (Sentiment140) is loaded and cleaned to remove noise like URLs, mentions and punctuation. Additionally, the tweets are tokenized and padded to ensure uniform input length for the model. Note: A precleaned dataset can also be loaded.
2. A CNN model is built with layers for embedding, convolution, pooling, and dense outputs. The embeddings are loaded pre-trained word embeddings from GloVe.
3. The model is compiled with an appropriate loss function and optimizer and trained on the pre-processed training data. This model is then saved for later use.
4. The model is then tested with a test dataset to determine its accuracy and loss.
5. A sample of 15 tweets are also selected for display to showcase the model's accuracy/results.
6. Support for single tweet sentiment analysis is also available.

# Model Architecture

The CNN model consists of:

Embedding Layer: Learns word representations from pre-trained GloVe embeddings.
Convolutional Layers: Extract features from sequences of word embeddings.
Max-Pooling Layer: Aggregates information from convolutional layers.
Dense Layers: Classify the sentiment as positive or negative.
Training Details

The model was trained for 10 epochs with a batch size of 128.
The Adam optimizer was used with a learning rate of 0.0005.
Class weights were applied to handle class imbalance in the dataset.
Early stopping was used to prevent overfitting.

# Results
The model achieved an accuracy of 82.3% on the test set.

# Sample Output
![sentimentmodelresults](https://github.com/user-attachments/assets/fbc0f964-1cc4-4ae6-b1f3-43e9240f775e)
