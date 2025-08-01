import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk_package_list = ['stopwords','punkt','punkt_tab','wordnet']

for pkg in nltk_package_list:
    nltk.download(pkg)

class TextPreprocessor:
    """
    This class handles the below:
    ● Normalize the text by making all letters lowercase.
    ● Remove all HTML tags (e.g., <br/>).
    ● Remove all email addresses.
    ● Remove all URLs.
    ● Remove all punctuation.
    ● Remove stop words.
    ● Lemmatize the words.
    """
    
    def __init__(self):
        pass

    def lower_case(self, sentence):
        return sentence.lower()

    def remove_html_tags(self, sentence):
        return re.sub(r'<[^<>]*>', '', sentence)

    def remove_email_addresses(self, sentence):
        return re.sub(r'\S*@\S*\s?', '', sentence)
    
    def remove_urls(self, sentence):
        return re.sub(r'www\.[^\s]+|http[^\s]+', '', sentence)

    def remove_puncts(self, sentence):
        return re.sub(r'[^\w\s]', '', sentence)

    def remove_stop_words(self, sentence):
        tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
        filtered = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(filtered)

    def lemattize_words(self, sentence):
        lemmatized = [self.lemmatizer.lemmatize(w) for w in self.w_tokenizer.tokenize(sentence)]
        return ' '.join(lemmatized)

    def preprocess_text(self, df):

        # Text cleaning
        df["review"] = df["review"].apply(self.lower_case)
        df["review"] = df["review"].apply(self.remove_html_tags)
        df["review"] = df["review"].apply(self.remove_email_addresses)
        df["review"] = df["review"].apply(self.remove_urls)
        df["review"] = df["review"].apply(self.remove_puncts)

        # Text preprocessing
        self.stop_words = set(stopwords.words('english'))
        df["review"] = df["review"].apply(self.remove_stop_words)

        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        df["review"] = df["review"].apply(self.lemattize_words)

        return df

 

    def save_pickle(self, df):
        output_dir="outputs"
        output_file="output_dataframe.pkl"        
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_pickle(f"{output_dir}/{output_file}")
        print("DataFrame successfully saved to {output_dir}/{output_file}")
    