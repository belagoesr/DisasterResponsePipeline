# Disaster Response Pipeline Project

The purpose of this project is to classify 

###Project Components
The project is divided in 3 parts:

1. ETL Pipeline
*$process_data.py*: write a data cleaning pipeline that loads the messages and categories datasets,
merges the two datasets, cleans the data and stores it in a SQLite database.

2. ML Pipeline
*$train_classifier.py*: write a machine learning pipeline that loads data from the SQLite database
and splits the dataset into training and test sets. Builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV and exports the final model as a pickle file.

3. Flask Web App
Show visualizations of the dataset and classify input a new message and get classification results in several categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        *$python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db*
    - To run ML pipeline that trains classifier and saves
        *$python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl*

2. Run the following command in the app's directory to run your web app.
    *$python run.py*

3. Go to http://0.0.0.0:3001/

