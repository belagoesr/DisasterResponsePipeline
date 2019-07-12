import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    receives two filepaths and return a dataframe of the 
    merged contents of filepath
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    receives a dataframe with a column category and return the dataframe cleaned with 
    each category in one column
    '''
    # split categories in diff columns
    categories = df['categories'].str.split(';', expand=True)
    # get categories names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # drop columns with all 0s
    categories = categories.drop(columns=['child_alone'])
    # convert categories to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].map(lambda x: 1 if int(x)!=0 else 0)
    # replace categories in original dataframe
    df = df.drop(columns=['categories'])
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates(subset=['id','message'])
    return df


def save_data(df, database_filename):
    '''
    save datataframe to database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)
  
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()