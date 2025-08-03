import kagglehub
import pandas as pd
from utils import TextPreprocessor
import os
from model import SVMModel, LSTMModel, GPTSentimentClassifier
import logging
import tensorflow as tf
import argparse
logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.DEBUG
 )


def import_data():
    # Download kaggle dataset
    logging.info("Downloading dataset")
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print("Path to dataset files:", path)

    # Read data
    df = pd.read_csv(path + "/IMDB Dataset.csv")
    return df

def preprocess_data(df):
    logging.info("Preprocess Data...")
    if os.path.exists("outputs/output_dataframe.pkl"):
        logging.info("Data already exists")
        df_processed = pd.read_pickle("outputs/output_dataframe.pkl")
        

    else:
        logging.info("Preprocessing Data...")
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_text(df)
        preprocessor.save_pickle(df_processed)
        logging.info("Preprocessing Done...")

    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df_processed['label'] = df_processed['sentiment'].map({'positive': 1, 'negative': 0})
    return df, df_processed


def call_model(model_type, df):
    logging.info("Calling model for training...")
    if model_type.lower() == "svm":
        model = SVMModel()
    elif model_type.lower() == "lstm":
        model = LSTMModel()
    
    X, y = model.tokenizer(df["review"], df["label"])
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    
    if type(model) is SVMModel:
        model.fit(X_train, y_train)
    elif type(model) is LSTMModel:
        logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    
    logging.info("Done training")
    return model, X_test, y_test

def eval_model(X_test, y_test):
    logging.info("Evaluating the model...")
    model.eval(X_test, y_test)

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models', 
        type=str, 
        choices=["svm", "lstm", "gpt2", "all"],
        required=True,
        help="Choose one or more models to run: 'svm', 'lstm', 'gpt2' or 'all'"
        )
    parser.add_argument(
        '--shots', 
        type=int, 
        default=1, 
        help='Specify number of shots for GPT2 model'
        )
    parser.add_argument(
        '--input_review', 
        type=str, 
        default="I found the movie extremely dull and uninspiring. Not worth the watch.", 
        help='Add input_review for gpt2'
        )


    df = import_data()

    # 1.Text Processing
    print("*"*20, " 1.Text Processing ", "*"*20)
    df, df_processed = preprocess_data(df)

    # 2. Machine Learning model for sentiment classification
    if "svm" in parser.models or "all" in parser.models:
        print("*"*20, " 2. Machine Learning model for sentiment classification ", "*"*20)
        model, X_test, y_test = call_model("svm", df_processed)
        eval_model(X_test, y_test)

    # 3.1. LSTM Model with preprocessed data
    if "lstm" in parser.models or "all" in parser.models:
        print("*"*20, " 3.1. LSTM Model with preprocessed data ", "*"*20)
        model, X_test, y_test = call_model("lstm", df_processed)
        eval_model(X_test, y_test)

    # 3.2. LSTM Model with original data
        print("*"*20, " 3.1. LSTM Model with preprocessed data ", "*"*20)
        model, X_test, y_test = call_model("lstm", df)
        eval_model(X_test, y_test)

    # 4. GPT-2 Classification using Prompt Engineering
    if "gpt2" in parser.models or "all" in parser.models:
        print("*"*20, " 4. GPT-2 Classification using Prompt Engineering ", "*"*20)
        gpt2_classifier = GPTSentimentClassifier(shots=1)
        
        review_input = parser.input_review
        sentiment = gpt2_classifier.classify(review_input, df)

        print(f"Predicted Sentiment: {sentiment}")    