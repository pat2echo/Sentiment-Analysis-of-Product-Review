import os
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "imblearn"])

import numpy as np
import pandas as pd
import re
from imblearn.over_sampling import SMOTE
import pickle

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

STUDENT_ID = 2320824
ASSIGNMENT_BASE_PATH = '/content/gdrive/MyDrive/CE807-24-SU/Reassessment'
ASSIGNMENT_BASE_PATH = '.'
model_gen = f'{ASSIGNMENT_BASE_PATH}/model/{STUDENT_ID}/Model_gen/'
model_dis = f'{ASSIGNMENT_BASE_PATH}/model/{STUDENT_ID}/Model_dis/'
output_dir = f'{ASSIGNMENT_BASE_PATH}/output/'
data_dir = f'{ASSIGNMENT_BASE_PATH}/data/'

class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text data using various machine learning models.

    This class provides functionality for:
    - Text preprocessing and cleaning
    - Feature extraction
    - Model training and evaluation
    - Dataset balancing
    - Model persistence

    Attributes:
        label_encoder (LabelEncoder): Encoder for converting sentiment labels to numerical values
        registration_number (int): Student ID for tracking
        reviews (list): Sample dataset for initial validation
        labels (list): Corresponding sentiment labels for the sample dataset
        lemmatizer (WordNetLemmatizer): Tool for word lemmatization
        stop_words (set): Collection of stop words to be removed during preprocessing
    """
    def __init__(self, student_id=1):
        """
        Initialize the SentimentAnalyzer with custom datasets and preprocessing tools.

        Args:
            student_id (int): Student registration number used as random seed
        """
        self.label_encoder = LabelEncoder()
        self.registration_number = student_id

        # Custom Created Dataset for first-level-validation
        self.reviews = [
            "This product is great, highly recommend!", "Terrible quality, do not buy.", "Good value for the price.",
            "Worst purchase I made.", "Amazing product, will buy again!", "Not worth the money.",
            "Great quality and fast shipping.", "I am so disappointed with this product.",
            "Perfect for what I needed, would buy again!", "Very poor quality, would not recommend."
        ]
        self.labels = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
          "positive", "negative"]

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Add domain-specific stopwords
        self.stop_words.update(['product', 'item', 'buy', 'purchased'])

    def count_text_features(self, series):
        """
        Counts the number of URLs, video tags, HTML tags,
        non-alphabetic characters, and emojis in a given text.

        Args:
            series: pd.Series text string to analyze.

        Returns:
            A dictionary containing the counts of each feature.
        """
        url_count = 0
        video_tag_count = 0
        html_tag_count = 0
        non_alphabetic_count = 0
        emoji_count = 0
        for text in series:
            if isinstance(text, str):
                url_count += len(
                    re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
                video_tag_count += len(re.findall(r'<video.*?>.*?</video>', text, flags=re.IGNORECASE))
                html_tag_count += len(re.findall(r'<[^>]+>', text))
                non_alphabetic_count += len(re.findall(r'[^a-zA-Z\s]', text))
                emoji_count += len(re.findall(r'[^\w\s]', text))  # Basic emoji detection (may need refinement)

        dict = {
            'url_count': url_count,
            'video_tag_count': video_tag_count,
            'html_tag_count': html_tag_count,
            'non_alphabetic_count': non_alphabetic_count,
            'emoji_count': emoji_count
        }
        print("Text Features Count: ", dict)
        print("+++++++++++++++++++")

    def clean_text(self, original_text):
        """
        Cleans the text in the given row for further processing.

        Args:
            original_text: String to be cleaned

        Returns:
            A Pandas Series with the cleaned text in the 'text' column.

        This function performs the following cleaning steps:
            1. Removes URLs.
            2. Removes video tags.
            3. Strips HTML tags.
            4. Removes non-alphabetic characters.
            5. Removes double spaces.
            6. Converts text to lowercase.
        """
        text = ''
        if isinstance(original_text, str):
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', original_text, flags=re.MULTILINE)
            # Remove video tags
            text = re.sub(r'(?i)\[\[video.*?\]\]', '', text)
            # Strip HTML tags
            text = re.sub(r'<br.*?>', '\n', text)
            # Strip HTML tags
            text = re.sub(r'<.*?>', ' ', text)

            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)

            # Remove double spaces
            text = re.sub(r'  ', ' ', text)

            # Convert to lower case
            text = text.lower()

        return text

    def truncate_text(self, tokens):
        """
        Truncates tokenized text to a maximum length.

        Args:
            tokens (list): List of text tokens

        Returns:
            list: Truncated list of tokens not exceeding the maximum word limit of 500
        """
        wc = len(tokens)
        max_words = 500
        if wc > max_words:
            tokens = tokens[:max_words]
        return tokens

    def remove_stopwords(self, tokens):
        """
        Removes stopwords from tokenized text.

        Args:
            tokens (list): List of text tokens

        Returns:
            list: Filtered list of tokens with stopwords removed
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
            row (pd.Series): Data row containing 'text' column

        Returns:
            pd.Series: Processed row with cleaned and tokenized text
        """
        text = row['text']

        # Clean text
        cleaned_text = self.clean_text(text)
        #cleaned_text = text

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Remove stopwords
        #tokens = self.remove_stopwords(tokens)

        # Lemmatize
        #tokens = self.lemmatize_text(tokens)

        # Join tokens back into text
        row['text'] = ' '.join(tokens)

        return row

    def balance_dataset(self, df):
        """
        Balances the dataset using SMOTE oversampling technique.

        Args:
            df (pd.DataFrame): Input DataFrame with 'text' and 'sentiment' columns

        Returns:
            pd.DataFrame: Balanced DataFrame with resampled data
        """
        # Convert text data to TF-IDF features
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=self.registration_number)
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['sentiment'])

        # Convert resampled TF-IDF features back to text
        inverse_transform = tfidf_vectorizer.inverse_transform(X_resampled)
        resampled_texts = [" ".join(tokens) for tokens in inverse_transform]

        # Create a new DataFrame with resampled data
        resampled_df = pd.DataFrame({'text': resampled_texts, 'sentiment': y_resampled})

        return resampled_df

    def get_train_data(self, train_file, val_file):
        """
        Loads and preprocesses training and validation data.

        Args:
            train_file: Path to the training data CSV file.
            val_file: Path to the validation data CSV file.

        Returns:
            A tuple containing:
                - train_df: Preprocessed training DataFrame.
                - val_df: Preprocessed validation DataFrame.

        This function performs the following steps:
            1. Loads training and validation data from CSV files.
            2. Drops rows with missing values (NaN).
            3. Applies text cleaning to the training data.
            4. Prints the number of records in each dataset.

        """
        train_df = pd.read_csv(train_file)
        self.count_text_features(train_df['text'])

        val_df = pd.read_csv(val_file)

        # train_data = train_df.sample(frac=0.1).reset_index(drop=True)  # Shuffling and selecting 10% of data
        #print(train_df[train_df.isna().any(axis=1)])
        train_df = train_df.dropna().apply(self.preprocess, axis=1)
        val_df = val_df.dropna().apply(self.preprocess, axis=1)
        print(train_df.head())
        print(f'Training data: {len(train_df)} records')
        print(f'Validation data: {len(val_df)} records')

        return train_df, val_df

    def get_test_data(self, test_file):
        """
        Loads and preprocesses testing data.

        Args:
            test_file: Path to the training data CSV file.

        Returns:
            A tuple containing:
                - test_file: Preprocessed training DataFrame.

        This function performs the following steps:
            1. Loads test data from CSV files.
            2. Drops rows with missing values (NaN).
            3. Applies text cleaning to the test data.
            4. Prints the number of records in each dataset.

        """
        test_df = pd.read_csv(test_file)
        test_df_o = test_df.copy()
        test_df = test_df.apply(self.preprocess, axis=1)
        print(test_df.head())
        print(f'Testing data: {len(test_df)} records')

        return test_df, test_df_o

    def gen_find_optimal_params(self, X_train, y_train, cv=5):
        """
        Performs grid search to find optimal parameters for CountVectorizer and MultinomialNB.

        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            dict: Dictionary of best parameters found during grid search
        """
        # Convert string labels to numerical values
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Calculate class prior probabilities from the training labels
        class_counts = np.bincount(y_train_encoded)
        class_priors = class_counts / len(y_train_encoded)  # Normalize to get probabilities
        print(f"Class Priors: {class_priors}")

        # Define the pipeline
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('nb', MultinomialNB(class_prior=class_priors))
        ])

        # Define parameter grid
        param_grid = {
            'vectorizer__max_features': [1000, 2000, 3000, 6000],
            'vectorizer__ngram_range': [(1, 2), (1, 3)],
            'vectorizer__min_df': [2, 3, 4],  # Minimum document frequency
            'nb__fit_prior': [True, False]  # Whether to learn class prior probabilities
        }
        param_grid = {
            'vectorizer__max_features': [3000],
            'vectorizer__ngram_range': [(1, 2)],
            'vectorizer__min_df': [4],  # Minimum document frequency
            'nb__fit_prior': [True]  # Whether to learn class prior probabilities
        }

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='f1_macro',
            verbose=1
        )

        print("***********************")
        print("******GRID SEARCH******")
        print("***********************")
        grid_search.fit(X_train, y_train_encoded)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        # Print top 3 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_3 = results_df.nlargest(3, 'mean_test_score')
        print("\nTop 3 parameter combinations:")
        for idx, row in top_3.iterrows():
            print(f"\nRank {row['rank_test_score']}:")
            print(f"Parameters: {row['params']}")
            print(f"Mean CV Score: {row['mean_test_score']:.3f} (+/- {row['std_test_score'] * 2:.3f})")

        return grid_search.best_params_

    def save_model(self, model_dir, model_name, model, vectorizer, label_encoder=None):
        """
        Saves a trained model and its associated vectorizer.

        Args:
            model_dir: Directory where the model and vocabulary files are saved.
            model_name: Name of the model.
            model: trained model
            vectorizer: trained vectorizer
            label_encoder: trained label_encoder
        Returns:
            None
        """
        model_file = os.path.join(model_dir, f'{model_name}_model.sav')
        pickle.dump(model, open(model_file, 'wb'))
        print('Saved model to', model_file)

        vocab_file = os.path.join(model_dir, f'{model_name}_vocab.sav')
        pickle.dump(vectorizer, open(vocab_file, 'wb'))
        print('Saved vocab to', vocab_file)

        label_encoder_file = os.path.join(model_dir, f'{model_name}_label_encoder.sav')
        pickle.dump(label_encoder, open(label_encoder_file, 'wb'))
        print('Saved label_encoder to', label_encoder_file)

    def load_model(self, model_dir, model_name):
        """
        Loads a saved model and its associated vectorizer.

        Args:
            model_dir: Directory where the model and vocabulary files are saved.
            model_name: Name of the model.

        Returns:
            A tuple containing the loaded model and vectorizer.
        """
        model_file = os.path.join(model_dir, f'{model_name}_model.sav')
        vocab_file = os.path.join(model_dir, f'{model_name}_vocab.sav')
        label_encoder_file = os.path.join(model_dir, f'{model_name}_label_encoder.sav')

        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f'Loaded model from {model_file}')

            with open(vocab_file, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f'Loaded vocabulary from {vocab_file}')

            with open(label_encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f'Loaded label_encoder from {label_encoder_file}')

            return model, vectorizer, label_encoder

        except FileNotFoundError:
            print(f"Error: Model files not found in {model_dir}.")
            return None, None, None

    def dis_find_optimal_params(self, X_train, y_train, cv=5):
        """
        Performs grid search to find optimal parameters for TfidfVectorizer and SVM.

        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            dict: Dictionary of best parameters found during grid search
        """
        # Convert string labels to numerical values
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Define the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svm', SVC(kernel='linear'))
        ])

        # Define parameter grid
        param_grid = {
            'tfidf__ngram_range': [(1, 2), (1, 3)],  # bigrams, trigrams
            'tfidf__max_features': [3000, 6000],  # Vocabulary size
            'tfidf__max_df': [0.7, 1.0],  # Maximum document frequency (proportion of documents)

            # SVM Parameters
            'svm__C': [0.1, 1, 10],  # Regularization parameter
            'svm__kernel': ['linear', 'rbf'],  # Kernel type
            #'svm__gamma': ['scale', 'auto'],  # Kernel coefficient (used for 'rbf')
            'svm__class_weight': [None, 'balanced'],  # Handle imbalanced classes
        }
        param_grid = {
            'tfidf__ngram_range': [(1, 3)],  # bigrams, trigrams
            'tfidf__max_features': [6000],  # Vocabulary size
            'tfidf__max_df': [0.7],  # Maximum document frequency (proportion of documents)

            # SVM Parameters
            'svm__C': [1],  # Regularization parameter
            'svm__kernel': ['linear'],  # Kernel type
            'svm__class_weight': ['balanced'],  # Handle imbalanced classes
        }


        print("***********************")
        print("******GRID SEARCH******")
        print("***********************")
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='f1_macro',  # Using f1_macro for multi-class
            verbose=1
        )

        grid_search.fit(X_train, y_train_encoded)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        # Print top 3 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_3 = results_df.nlargest(3, 'mean_test_score')
        print("\nTop 3 parameter combinations:")
        for idx, row in top_3.iterrows():
            print(f"\nRank {row['rank_test_score']}:")
            print(f"Parameters: {row['params']}")
            print(f"Mean CV Score: {row['mean_test_score']:.3f} (+/- {row['std_test_score'] * 2:.3f})")

        return grid_search.best_params_

    def evaluation_results(self, y_test, y_pred):
        """
        Evaluates model performance using various metrics.

        Args:
            y_test (array-like): True labels
            y_pred (array-like): Predicted labels

        Prints:
            DataFrame containing comparison of model performance against positive and negative benchmarks,
            including metrics for:
            - Accuracy
            - F1 Score (Macro)
            - Recall (Macro)
            - Precision (Macro)
        """
        # Print the accuracy and detailed classification report
        positive_benchmark = ["positive"] * len(y_pred)
        negative_benchmark = ["negative"] * len(y_pred)
        metrics_data = {
            "Metric": ["Accuracy", "F1 Score (Macro)", "Recall (Macro)", "Precision (Macro)"],
            "Model": [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                precision_score(y_test, y_pred, average="macro", zero_division=True),
            ],
            "Positive Benchmark": [
                accuracy_score(y_test, positive_benchmark),
                f1_score(y_test, positive_benchmark, average="macro"),
                recall_score(y_test, positive_benchmark, average="macro"),
                precision_score(y_test, positive_benchmark, average="macro", zero_division=True),
            ],
            "Negative Benchmark": [
                accuracy_score(y_test, negative_benchmark),
                f1_score(y_test, negative_benchmark, average="macro"),
                recall_score(y_test, negative_benchmark, average="macro"),
                precision_score(y_test, negative_benchmark, average="macro", zero_division=True),
            ],
        }

        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        print("*******************")
        print("******RESULTS******")
        print("*******************")
        print(metrics_df)

def train_Gen(train_file, val_file, model_dir, student_id=2320824):
    """
    Trains a Naive Bayes model for sentiment classification.

    Args:
        train_file: Path to the training data CSV file.
        val_file: Path to the validation data CSV file.

    This function performs the following steps:
        1. Loads and preprocesses training and validation data using `get_train_data`.
        2. Converts text data to feature vectors using CountVectorizer.
        3. Splits the training data into training and testing sets.
        4. Trains a Multinomial Naive Bayes model.
        5. Evaluates the model's performance on the testing set using classification report.
        6. Prints the evaluation results.

    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("*************************************")
    print("----GENERATIVE MODEL: NAIVE BAYES----")
    print("*************************************")
    gen = SentimentAnalyzer(student_id=student_id)

    # Get train and validation data
    train_df, val_df = gen.get_train_data(train_file, val_file)
    return
    #train_df = gen.balance_dataset(train_df_unbal)

    # Split into train and test sets
    X_train = train_df['text']
    X_test = val_df['text']
    y_train = train_df['sentiment']
    y_test = val_df['sentiment']

    # Find optimal parameters
    best_params = gen.gen_find_optimal_params(X_train, y_train)

    # Extract best parameters
    vectorizer_params = {k.replace('vectorizer__', ''): v
                         for k, v in best_params.items()
                         if k.startswith('vectorizer__')}
    nb_params = {k.replace('nb__', ''): v
                 for k, v in best_params.items()
                 if k.startswith('nb__')}

    print("\nTraining final model with:")
    print(f"Vectorizer parameters: {vectorizer_params}")
    print(f"Naive Bayes parameters: {nb_params}")

    # Initialize vectorizer with optimal parameters
    vectorizer = CountVectorizer(**vectorizer_params)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Encode labels
    y_train_encoded = gen.label_encoder.fit_transform(y_train)

    # Train the model with optimal parameters
    model = MultinomialNB(**nb_params)
    model.fit(X_train_vec, y_train_encoded)

    # save model
    gen.save_model(model_dir=model_dir, model_name='naive_bayes', model=model, vectorizer=vectorizer, label_encoder=gen.label_encoder)

    # Evaluate the model
    y_pred_encoded = model.predict(X_test_vec)
    y_pred = gen.label_encoder.inverse_transform(y_pred_encoded)
    gen.evaluation_results(y_test, y_pred)

    return vectorizer, model

train_Gen(f'{data_dir}train.csv', f'{data_dir}valid.csv', model_gen, student_id=STUDENT_ID)

def test_Gen(test_file, model_dir, student_id=2320824):

    if not os.path.exists(model_dir):
        print(f"Error: Model dir {model_dir} not found.")

    print("------------------------------------------")
    print("----TEST GENERATIVE MODEL: NAIVE BAYES----")
    print("------------------------------------------")
    gen = SentimentAnalyzer(student_id=student_id)

    # Get test data and apply preprocessing rules
    test_df, test_df_o = gen.get_test_data(test_file)

    # load model
    model, vectorizer, label_encoder = gen.load_model(model_dir=model_dir, model_name='naive_bayes')

    # Initialize vectorizer
    X_test_vec = vectorizer.transform(test_df['text'])

    # Evaluate the model
    y_pred_encoded = model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    #gen.evaluation_results(test_df['sentiment'], y_pred)
    test_df_o['out_label_model_gen'] = y_pred
    test_df_o.to_csv(f'{model_dir}text.csv', index=False)
    test_df_o.to_csv(test_file, index=False)
    print("Saved Generative test results to:", f'{model_dir}text.csv')

    return 'out_label_model_gen', y_pred

#test_Gen(f'{data_dir}test.csv', model_gen, student_id=STUDENT_ID)

def train_Dis(train_file, val_file, model_dir, student_id=2320824):
    """
    Trains a Support Vector Machine (SVM) model for sentiment classification.

    Args:
        train_file: Path to the training data CSV file.
        val_file: Path to the validation data CSV file.

    This function performs the following steps:
        1. Loads and preprocesses training and validation data using `get_train_data`.
        2. Splits the training data into training and testing sets.
        3. Extracts features using TF-IDF vectorizer.
        4. Trains a linear SVM model.
        5. Evaluates the model's performance on the testing set using classification report.
        6. Prints the evaluation results.

    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("*********************************")
    print("----DISCRIMINATIVE MODEL: SVM----")
    print("*********************************")
    dis = SentimentAnalyzer(student_id=student_id)
    train_df_unbal, val_df = dis.get_train_data(train_file, val_file)
    train_df = dis.balance_dataset(train_df_unbal)

    # Split into train and test sets
    X_train = train_df['text']
    X_test = val_df['text']
    y_train = train_df['sentiment']
    y_test = val_df['sentiment']

    # Find optimal parameters
    best_params = dis.dis_find_optimal_params(X_train, y_train)

    # Extract best parameters
    ngram_range = best_params['tfidf__ngram_range']
    max_features = best_params['tfidf__max_features']

    print(f"Training final model with ngram_range={ngram_range}, max_features={max_features}")

    # Initialize vectorizer with optimal parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=ngram_range
    )

    # Transform the text data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Encode labels
    y_train_encoded = dis.label_encoder.fit_transform(y_train)

    # Train the SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train_encoded)

    # save the model
    dis.save_model(model_dir=model_dir, model_name='svm', model=svm_model, vectorizer=vectorizer, label_encoder=dis.label_encoder)

    # Evaluate the model
    y_pred_encoded = svm_model.predict(X_test_tfidf)

    # Convert predictions back to original labels for evaluation
    y_pred = dis.label_encoder.inverse_transform(y_pred_encoded)
    dis.evaluation_results(y_test, y_pred)

    return vectorizer, svm_model

#train_Dis(f'{data_dir}train.csv', f'{data_dir}valid.csv', model_dis, student_id=STUDENT_ID)

def test_Dis(test_file, model_dir, student_id=2320824):

    if not os.path.exists(model_dir):
        print(f"Error: Model dir {model_dir} not found.")

    print("--------------------------------------")
    print("----TEST DISCRIMINATIVE MODEL: SVM----")
    print("--------------------------------------")
    dis = SentimentAnalyzer(student_id=student_id)

    # Get test data and apply preprocessing rules
    test_df, test_df_o = dis.get_test_data(test_file)

    # load model
    model, vectorizer, label_encoder = dis.load_model(model_dir=model_dir, model_name='svm')

    # Initialize vectorizer
    X_test_vec = vectorizer.transform(test_df['text'])

    # Evaluate the model
    y_pred_encoded = model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    #dis.evaluation_results(test_df['sentiment'], y_pred)

    test_df_o['out_label_model_dis'] = y_pred
    test_df_o.to_csv(f'{model_dir}text.csv', index=False)
    test_df_o.to_csv(test_file, index=False)
    print("Saved Discriminative test results to:", f'{model_dir}text.csv')

    return 'out_label_model_dis', y_pred

#test_Dis(f'{data_dir}test.csv', model_dis, student_id=STUDENT_ID)