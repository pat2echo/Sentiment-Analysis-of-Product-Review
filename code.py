from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

    def generative_model(self):
        print("*************************************")
        print("----GENERATIVE MODEL: NAIVE BAYES----")
        print("*************************************")
        # Convert text to feature vectors
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.reviews)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, self.labels, test_size=0.2, random_state=self.registration_number)

        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        self.evaluation_results(y_test, y_pred)

    def discriminative_model(self):
        print("*********************************")
        print("----DISCRIMINATIVE MODEL: SVM----")
        print("*********************************")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.reviews, self.labels, test_size=0.2,
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
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0, output_dict=False))

sa = SentimentAnalyzer()
sa.generative_model()
sa.discriminative_model()