import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def get_plot_data(dataframe):
    # data for first plot
    categories_count = dataframe.drop(columns=['original','message','id']).melt(id_vars=['genre'])\
                        .groupby(by='variable').sum().sort_values(by='value', ascending=True)
    x1 = categories_count.value.index.str.capitalize()
    y1 = categories_count.value.values

    # data for second plot
    genre_count = dataframe.genre.value_counts()
    x2 = genre_count.index.str.capitalize()
    y2 = genre_count.values
    
    return [(x1,y1), (x2,y2)]
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # get data to plot
    plot_data = get_plot_data(df)
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    y=plot_data[0][0],
                    x=plot_data[0][1],
                    marker = {
                        'color': '#414770', 
                        'line': { 
                            'color': 'rgb(0,0,0)',
                            'width': 2.0
                        }
                    },
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of messages labels',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Count"
                },
                'width': 1100,
                'height': 600,
                'margin': {'l': 250, 'r': 100, 'b': 100, 't': 50},
                'paper_bgcolor': 'rgb(248, 248, 255)',
                'plot_bgcolor': 'rgb(248, 248, 255)',
            }
        },
        {
            'data': [
                Bar(
                    x=plot_data[1][0],
                    y=plot_data[1][1],
                    marker = {
                        'color': '#414770', 
                        'line': { 
                            'color': 'rgb(0,0,0)',
                            'width': 2.0
                        }
                    },
                )
            ],

            'layout': {
                'title': 'Distribution of messages types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message type"
                },
                'width': 1100,
                'height': 500,
                'margin': {'l': 250, 'r': 250, 'b': 100, 't': 50},
                'paper_bgcolor': 'rgb(248, 248, 255)',
                'plot_bgcolor': 'rgb(248, 248, 255)',
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    c1 = classification_labels[0:12]
    c2 = classification_labels[12:24]
    c3 = classification_labels[24:]
    labels_1 = df.columns[4:][0:16]
    labels_2 = df.columns[4:][16:]
    labels_3 = df.columns[4:][16:]
    #classification_results = dict(zip(df.columns[4:], classification_labels))
    classification_results_1 = dict(zip(labels_1, c1))
    classification_results_2 = dict(zip(labels_2, c2))
    classification_results_3 = dict(zip(labels_2, c3))
   

    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        #classification_result=classification_results
        classification_result_1 = classification_results_1,
        classification_result_2 = classification_results_2,
        classification_result_3 = classification_results_3,
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()