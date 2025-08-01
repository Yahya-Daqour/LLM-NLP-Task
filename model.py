import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


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

        # Metrics
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        # Reports
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")


class SVM(SentimentClassifierModel):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LinearSVC()

    def tokenizer (self, review_text, sentiment):
        X = self.vectorizer.fit_transform(review_text)
        y = sentiment
        return X, y

