import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

class SentimentAnalyzer:
    def __init__(self):
        self.registration_number = 2320824
        self.reviews = [
            "This product is great, highly recommend!", "Terrible quality, do not buy.", "Good value for the price.",
            "Worst purchase I made.", "Amazing product, will buy again!", "Not worth the money.",
            "Great quality and fast shipping.", "I am so disappointed with this product.",
            "Perfect for what I needed, would buy again!", "Very poor quality, would not recommend."
        ]
        self.labels = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
          "positive", "negative"]

    def clean_text(self, row):
        # Preserve original text
        original_text = row['text']

        if isinstance(original_text, str):
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', original_text, flags=re.MULTILINE)
            # Remove video tags
            # text = re.sub(r'\[video.*?\].*?\[/video\]', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)\[\[video.*?\]\]', '', text)
            # Strip HTML tags
            text = re.sub(r'<br.*?>', '\n', text)
            # Strip HTML tags
            text = re.sub(r'<.*?>', ' ', text)

            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Remove double spaces
            text = re.sub(r'  ', ' ', text)
            # Convert to lower case
            text = text.lower()

            # Truncate text to the maximum number of words that the model can handle
            tokens = text.split()
            wc = len(tokens)
            max_words = 200
            if wc > max_words:
                text = ' '.join(tokens[:max_words])

            row['text'] = text

        return row

    def get_train_data(self, train_file, val_file):
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        # train_data = train_df.sample(frac=0.1).reset_index(drop=True)  # Shuffling and selecting 10% of data
        #print(train_df[train_df.isna().any(axis=1)])
        print(train_df.head())
        train_df = train_df.dropna().apply(self.clean_text, axis=1)
        print(f'Training data: {len(train_df)} records')
        print(f'Validation data: {len(val_df)} records')

        return train_df, val_df

    def train_gen(self, train_file, val_file):
        print("*************************************")
        print("----GENERATIVE MODEL: NAIVE BAYES----")
        print("*************************************")

        train_df, val_df = self.get_train_data(train_file, val_file)
        #return

        # Convert text to feature vectors
        # Note no pre-processing is done here. In practice you will have different preprocessing steps.
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(train_df['text'])

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, train_df['sentiment'], test_size=0.2, random_state=self.registration_number)

        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        self.evaluation_results(y_test, y_pred)

    def train_dis(self, train_file, val_file):
        print("*********************************")
        print("----DISCRIMINATIVE MODEL: SVM----")
        print("*********************************")
        train_df, val_df = self.get_train_data(train_file, val_file)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['sentiment'], test_size=0.2,
                                                            random_state=self.registration_number)

        # Step 2: Feature Extraction using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english',
                                     max_features=1000)  # Using TF-IDF to convert text to features
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Step 3: Train the SVM model
        svm_model = SVC(kernel='linear')  # Using a linear kernel for text classification
        svm_model.fit(X_train_tfidf, y_train)

        # Step 4: Evaluate the model
        y_pred = svm_model.predict(X_test_tfidf)
        self.evaluation_results(y_test, y_pred)


    def evaluation_results(self, y_test, y_pred):
        # Print the accuracy and detailed classification report
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0, output_dict=False))

gen = SentimentAnalyzer()
gen.train_gen(train_file='./data/train.csv', val_file='./data/valid.csv')

dis = SentimentAnalyzer()
dis.train_dis(train_file='./data/train.csv', val_file='./data/valid.csv')