from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentClassifierModel:
    def __init__(self):
        pass 

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def eval(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)


        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")


class SVMModel(SentimentClassifierModel):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LinearSVC()

    def tokenizer(self, review_text, sentiment):
        X = self.vectorizer.fit_transform(review_text)
        y = sentiment
        return X, y
    
class LSTMModel(SentimentClassifierModel):
    def __init__(self, vocab_size=10000, max_len=200, embedding_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

    def tokenizer(self, review, sentiment):
        self.tokenizer_.fit_on_texts(review)
        sequences = self.tokenizer_.texts_to_sequences(review)
        X = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        y = np.array(sentiment)
        return X, y
    
    def build_model(self):
        embedding = Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)
        model = Sequential([
            embedding,
            LSTM(128),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model

    def fit(self, X_train, y_train, batch_size=64, epochs=5, validation_data=None):
        if self.model is None:
            self.build_model()
        self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=2
        )
    
    def eval(self, X_test, y_test):
        y_probs = self.model.predict(X_test).flatten()
        y_pred = (y_probs >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")       
