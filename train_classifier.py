import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """ INPUT:
        database_filepath: filepath to the database

        OUTPUT:
        - X: DataFrame with the text of each message in english
        - Y: DataFrame with the right categorization to each text message in a
        form that sklearn can work (with dummy variables for each category)
        - category_names: list of the possible categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = engine.table_names()[0]
    df = pd.read_sql_table(table_name, engine)
    X = df.message
    Y = df[[column for column in df.columns if column\
        not in ['id', 'message', 'original', 'genre']]]

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """ This function is used as part of the count vectorizer to create tokens
        out of plain text

        INPUT:
        - text: string
        OUTPUT:
        - tokens: normalization and tokenization of the given string
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower().strip()
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word\
              not in stopwords.words('english')]
    return tokens

def build_model():
    """ Build the pipeline for the text analysis and get it ready for fitting

        OUTPUT:
        model: pipeline ready for fitting
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    model = pipeline

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Get metrics of the model

        INPUT:
        - model: model already fitted
        - X_test, Y_test: test dataframe for cross validation
        - category_names: names of each category

        OUTPUT:
        - the accuracy, precision, recall and f1-score for each category
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=Y_pred, columns=category_names)

    for column in Y_test.columns:
        metrics = classification_report(Y_test[column], Y_pred[column])
        print('Metrics for the category "{}" are:\n{}\n'.format(column, metrics))


def save_model(model, model_filepath):
    joblib.dump(model, 'model_filepath')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
