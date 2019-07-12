import sys
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score, classification_report, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download(['punkt','wordnet','stopwords'])

def load_data(database_filepath):
    '''
    reads the database contained in filepath and save it in a dataframe,
    returns X, y and labels of columns
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.iloc[:,4:].values
    labels = df.columns[4:]
    return X, y, labels

def tokenize(text):
    '''
    receive text messages and return list with tokens lemmatized and stemmed
    '''
    # Normalization
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
 
    # Tokenize
    words = nltk.word_tokenize(text)
 
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w).strip() for w in words]
    
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    return stemmed

def build_model():
    '''
    return the Pipeline with the steps to be performed on the dataframe
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(loss='hinge', penalty='l2'))),
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    '''
    receive the model and print the precision, recall and accuracy for each category
    '''
    y_preds = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i].upper())
        acc = accuracy_score(y_test[:, i], y_preds[:, i])
        recall = recall_score(y_test[:, i], y_preds[:, i], average='weighted')
        prec = precision_score(y_test[:, i], y_preds[:, i], average='weighted')
        f1 = f1_score(y_test[:, i], y_preds[:, i], average='weighted')
        print('[{0}]Accuracy:{1:.2f} [{0}]Recall:{2:.2f} [{0}]Precision:{3:.2f} [{0}]F1_score:{4:.2f}\n' \
          .format('SVM', acc, recall, prec, f1))

    

def save_model(model, model_filepath):
    '''
    receive the model and save it into a pickle file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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