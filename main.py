import kagglehub
import pandas as pd
from utils import TextPreprocessor
import os
from model import SVM
import logging

logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.DEBUG
 )

# Download kaggle dataset
logging.info("Downloading dataset")
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)

# Read data
df = pd.read_csv(path + "/IMDB Dataset.csv")
print(df)

# Preprocess
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

# Call model
logging.info("Calling model for training...")
svm_model = SVM()
X,y = svm_model.tokenizer(df_processed["review"], df_processed["sentiment"])
X_train, X_test, y_train, y_test = svm_model.split_data(X, y)
svm_model.fit(X_train, y_train)
logging.info("Done training")

# Evaluate
logging.info("Evaluating the model...")
svm_model.eval(X_test, y_test)