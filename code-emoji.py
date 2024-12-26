import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "imblearn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "emoji"])

import pandas as pd
import numpy as np
import re
import pickle
import emoji
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SentimentAnalyzer:
    """
    A comprehensive class for sentiment analysis with emoji support.
    """

    def __init__(self, student_id=1):
        """
        Initialize the SentimentAnalyzer with custom datasets and preprocessing tools.

        Args:
            student_id (int): Student identification number for tracking purposes
        """
        self.label_encoder = LabelEncoder()
        self.registration_number = student_id

        # Custom Created Dataset for first-level-validation
        self.reviews = [
            "This product is great, highly recommend!", "Terrible quality, do not buy.",
            "Good value for the price.", "Worst purchase I made.",
            "Amazing product, will buy again!", "Not worth the money.",
            "Great quality and fast shipping.", "I am so disappointed with this product.",
            "Perfect for what I needed, would buy again!", "Very poor quality, would not recommend."
        ]
        self.labels = ["positive", "negative", "positive", "negative", "positive",
                       "negative", "positive", "negative", "positive", "negative"]

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['product', 'item', 'buy', 'purchased'])

        # Add emoji mapping for sentiment
        self.emoji_sentiment = {
            '😊': 'positive', '😃': 'positive', '😄': 'positive', '👍': 'positive',
            '❤️': 'positive', '😍': 'positive', '🎉': 'positive', '✨': 'positive',
            '😢': 'negative', '😞': 'negative', '😠': 'negative', '👎': 'negative',
            '😡': 'negative', '💔': 'negative', '😒': 'negative', '😩': 'negative'
        }

    def extract_emojis(self, text):
        """
        Extracts emojis from text and returns them as a list.

        Args:
            text (str): Input text containing emojis

        Returns:
            list: List of emojis found in the text
        """
        return [c for c in text if c in emoji.EMOJI_DATA]

    def get_emoji_sentiment_score(self, emojis):
        """
        Calculates sentiment score based on emojis.

        Args:
            emojis (list): List of emojis from text

        Returns:
            float: Sentiment score between -1 and 1
        """
        score = 0
        for e in emojis:
            if e in self.emoji_sentiment:
                score += 1 if self.emoji_sentiment[e] == 'positive' else -1
        return score / len(emojis) if emojis else 0

    def extract_text_features(self, text):
        """
        Extracts various features from text including emoji-related features.

        Args:
            text (str): Input text

        Returns:
            dict: Dictionary containing text features
        """
        emojis = self.extract_emojis(text)
        emoji_count = len(emojis)
        emoji_sentiment = self.get_emoji_sentiment_score(emojis)

        url_count = len(
            re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        video_tag_count = len(re.findall(r'<video.*?>.*?</video>', text, flags=re.IGNORECASE))
        html_tag_count = len(re.findall(r'<[^>]+>', text))

        return {
            'url_count': url_count,
            'video_tag_count': video_tag_count,
            'html_tag_count': html_tag_count,
            'emoji_count': emoji_count,
            'emoji_sentiment': emoji_sentiment,
            'emojis': Counter(emojis)
        }

    def clean_text(self, original_text):
        """
        Cleans and normalizes text data while preserving emojis.

        Args:
            original_text (str): Raw text input to be cleaned

        Returns:
            str: Cleaned text with emojis preserved
        """
        if not isinstance(original_text, str):
            return ''

        # Extract emojis before cleaning
        emojis = self.extract_emojis(original_text)

        # Clean text but preserve emojis
        text = original_text

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove video tags
        text = re.sub(r'(?i)\[\[video.*?\]\]', '', text)

        # Strip HTML tags
        text = re.sub(r'<br.*?>', '\n', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Remove non-alphabetic characters except emojis
        text = ''.join(c for c in text if c.isalpha() or c.isspace() or c in emoji.EMOJI_DATA)

        # Remove double spaces
        text = re.sub(r'\s+', ' ', text)

        # Convert to lower case (doesn't affect emojis)
        text = text.lower()

        return text.strip()

    def truncate_text(self, tokens):
        """
        Truncates tokenized text to a maximum length.

        Args:
            tokens (list): List of text tokens

        Returns:
            list: Truncated list of tokens
        """
        max_words = 500
        return tokens[:max_words] if len(tokens) > max_words else tokens

    def remove_stopwords(self, tokens):
        """
        Removes stopwords from tokenized text.

        Args:
            tokens (list): List of text tokens

        Returns:
            list: Filtered list of tokens
        """
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize_text(self, tokens):
        """
        Performs lemmatization on tokenized text.

        Args:
            tokens (list): List of text tokens

        Returns:
            list: List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, row):
        """
        Applies complete preprocessing pipeline to a data row.

        Args:
            row (pd.Series): Data row containing text

        Returns:
            pd.Series: Processed row with additional features
        """
        text = row['text']

        # Extract features before cleaning
        features = self.extract_text_features(text)

        # Clean text while preserving emojis
        cleaned_text = self.clean_text(text)

        # Tokenize (including emojis)
        tokens = word_tokenize(cleaned_text)

        # Optional steps (commented out by default)
        # tokens = self.remove_stopwords(tokens)
        # tokens = self.lemmatize_text(tokens)
        tokens = self.truncate_text(tokens)

        # Add emoji features to row
        row['emoji_count'] = features['emoji_count']
        row['emoji_sentiment'] = features['emoji_sentiment']

        # Join tokens back into text
        row['text'] = ' '.join(tokens)

        return row

    def balance_dataset(self, df):
        """
        Balances the dataset using SMOTE oversampling technique.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

        smote = SMOTE(random_state=self.registration_number)
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['sentiment'])

        inverse_transform = tfidf_vectorizer.inverse_transform(X_resampled)
        resampled_texts = [" ".join(tokens) for tokens in inverse_transform]

        return pd.DataFrame({'text': resampled_texts, 'sentiment': y_resampled})

    def create_feature_matrix(self, texts, vectorizer=None, training=False):
        """
        Creates a feature matrix combining text vectors and emoji features.

        Args:
            texts (list): List of text samples
            vectorizer: Optional pre-fitted vectorizer
            training (bool): Whether this is for training

        Returns:
            tuple: (combined_features, vectorizer)
        """
        # Extract emoji features
        emoji_features = []
        for text in texts:
            features = self.extract_text_features(text)
            emoji_features.append([
                features['emoji_count'],
                features['emoji_sentiment']
            ])
        emoji_features = np.array(emoji_features)

        # Create text features
        if training:
            if vectorizer is None:
                vectorizer = TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 3),
                    max_df=0.7
                )
            text_features = vectorizer.fit_transform(texts)
        else:
            text_features = vectorizer.transform(texts)

        # Combine features
        text_features_dense = text_features.toarray()
        combined_features = np.hstack((text_features_dense, emoji_features))

        return combined_features, vectorizer

    def get_train_data(self, train_file, val_file):
        """
        Loads and preprocesses training and validation data.

        Args:
            train_file (str): Path to training file
            val_file (str): Path to validation file

        Returns:
            tuple: (train_df, val_df)
        """
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        # Add emoji features columns
        for df in [train_df, val_df]:
            df['emoji_count'] = 0
            df['emoji_sentiment'] = 0.0

        train_df = train_df.dropna().apply(self.preprocess, axis=1)
        val_df = val_df.dropna().apply(self.preprocess, axis=1)

        print(f'Training data: {len(train_df)} records')
        print(f'Validation data: {len(val_df)} records')

        return train_df, val_df

    def get_test_data(self, test_file):
        """
        Loads and preprocesses test data.

        Args:
            test_file (str): Path to test file

        Returns:
            tuple: (preprocessed_df, original_df)
        """
        test_df = pd.read_csv(test_file)
        test_df_original = test_df.copy()

        test_df['emoji_count'] = 0
        test_df['emoji_sentiment'] = 0.0

        test_df = test_df.apply(self.preprocess, axis=1)

        print(f'Testing data: {len(test_df)} records')

        return test_df, test_df_original

    def train_model(self, train_df, val_df, model_dir, model_name):
        """
        Trains a sentiment analysis model with emoji features.

        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            model_dir (str): Directory to save model
            model_name (str): Name for saved model

        Returns:
            tuple: (model, vectorizer, metrics)
        """
        print("Starting model training with emoji features...")

        X_train = train_df['text'].values
        y_train = train_df['sentiment'].values
        X_val = val_df['text'].values
        y_val = val_df['sentiment'].values

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        X_train_combined, vectorizer = self.create_feature_matrix(X_train, training=True)
        X_val_combined, _ = self.create_feature_matrix(X_val, vectorizer=vectorizer, training=False)

        model = SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            random_state=self.registration_number
        )
        model.fit(X_train_combined, y_train_encoded)

        y_train_pred = model.predict(X_train_combined)
        y_val_pred = model.predict(X_val_combined)

        y_train_pred = self.label_encoder.inverse_transform(y_train_pred)
        y_val_pred = self.label_encoder.inverse_transform(y_val_pred)

        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'f1_macro': f1_score(y_train, y_train_pred, average='macro'),
                'precision_macro': precision_score(y_train, y_train_pred, average='macro'),
                'recall_macro': recall_score(y_train, y_train_pred, average='macro')
            },
            'validation': {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'f1_macro': f1_score(y_val, y_val_pred, average='macro'),
                'precision_macro': precision_score(y_val, y_val_pred, average='macro'),
                'recall_macro': recall_score(y_val, y_val_pred, average='macro')
            }
        }

        print(metrics)

        return model, vectorizer, metrics


gen = SentimentAnalyzer(student_id=2320824)
tdf, vdf = gen.get_train_data(train_file=f'./data/train.csv', val_file=f'./data/valid.csv')
gen.train_model(tdf, vdf, '', 'svm')