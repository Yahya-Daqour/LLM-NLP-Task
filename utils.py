import pandas as pd
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

    def lower_case(sentence):
        return sentence.lower()

    def remove_html_tags(sentence):
        return re.sub(r'<[^<>]*>', '', sentence)

    def remove_email_addresses(sentence):
        return re.sub(r'\S*@\S*\s?', '', sentence)
    
    def remove_urls(sentence):
        return re.sub(r'www\.[^\s]+|http[^\s]+', '', sentence)

    def remove_puncts(sentence):
        return re.sub(r'[^\w\s]', '', sentence)

    def remove_stop_words(sentence):
        tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
        filtered = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(filtered)

    def lemattize_words(sentence):
        lemmatized = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(sentence)]
        return ' '.join(lemmatized)

    def preprocess_text(df):

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

 

    def save_pickle():
        pass
    






# Remove stop words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Sample text
text = "This is a sample sentence showing stopword removal."

# Get English stopwords and tokenize
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text.lower())

# Remove stopwords
filtered_tokens = [word for word in tokens if word not in stop_words]

print("Original:", tokens)
print("Filtered:", filtered_tokens)



# Download required NLTK data (only needed once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from a single sentence
def remove_stopwords(sentence):
    tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

# Apply to the 'text' column
df['review'] = df['review'].apply(remove_stopwords)

# View result
df

# Lemmatize the words.


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
     lemmatized = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
     return ' '.join(lemmatized)

df['review'] = df.review.apply(lemmatize_text)

df.to_csv("tmp.csv")