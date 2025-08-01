import kagglehub
import pandas as pd
from utils import TextPreprocessor

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)




df = pd.read_csv(path + "/IMDB Dataset.csv")
print(df)


preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_text(df)
preprocessor.save_pickle(df_processed)
