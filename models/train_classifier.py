# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


import pickle


def load_data(database_filepath):
    '''
        INPUT:
        database_filepath - this is the database file where we stored the merged data of messages and their categories

        OUTPUT:
        X - array of messages
        Y - array of message labels
        category_names - list of label names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages_categorised", engine)
    #Remove related =2 because messages are in foreign language for some. Not interpretable and have no labels for the label variables
    df = df[df.related != 2].reset_index(drop=True)

    X = df.message.values

    Y = df[['related', 'request', 'offer',
            'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']]

    Y.drop(columns='child_alone', inplace=True)

    category_names = list(Y.columns)
    return X,Y, category_names


def tokenize(text):
    '''
        INPUT:
        text - message that we want to normalise, tokenize and lemmatise

        OUTPUT:
        lemmed - list of processed words
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Replace url link if available
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, 'urlplaceholder')

    text = text.lower()
    # Remove punctuations
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)

    # Remove stop words
    words = [x for x in tokens if x not in stopwords.words("english")]

    # Stem the words - reduce word to root form
    #         stemmed = [PorterStemmer().stem(w) for w in words]
    # Lemmatise words - reduce word to root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


# In our dataset, the labels are unevenly distributed.
# Labels such as 'offer', 'security', 'clothing', 'missing_people', 'electricity', 'tools', 'hospitals', 'shops',
# 'aid_centers', 'fire', 'cold' are used on around 500 or less messages out of over 26,000 messages.
# This reduces the effectiveness of our machine learning algorithm when training the model on these minority labels.
# 3. To overcome this data imbalance, we use data augmentation by oversampling these minority labels. We leverage the technique
# called multi-label synthetic minority over-sampling (MLSMOTE).

def create_dataset(n_sample=1000):
    '''
    Create a unevenly distributed sample data set multilabel
    classification using make_classification function

    args
    nsample: int, Number of sample to be created

    return
    X: pandas.DataFrame, feature vector dataframe with 10 features
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2,
                               weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y


def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe

    args
    df: pandas.DataFrame, target label df whose tail label has to identified

    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def get_index(df):
    """
    give the index of all tail_label rows
    args
    df: pandas.DataFrame, target label df from which index for tail label has to identified

    return
    index: list, a list containing index number of all the tail label
    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)
        index = index.union(sub_index)
    return list(index)


def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels

    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe

    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target

def augment_data(X,Y):
    '''
    Before augmenting the data, we create a pipeline of CountVectorizer, TfidfTransformer to transform the text message into a vector numbers
    Then we obtain the minority labels from the data set - X_sub, Y_sub
    We generate extra observations for these minority labels - X_res, Y_res
    We append the extra observations to the original dataset and transform them into arrays - X_array, Y_array

     RandomForestClassifier.
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer())])
    pipeline.fit(X.ravel())
    X_transf = pipeline.transform(X.ravel())
    X_transf_df = pd.DataFrame.sparse.from_spmatrix(X_transf)

    X_sub, Y_sub = get_minority_instace(X_transf_df, Y)

    X_res, Y_res = MLSMOTE(X_sub, Y_sub, 1000)

    X_transf_df = X_transf_df.append(X_res)
    Y = Y.append(Y_res)
    Y_array = Y.values
    X_array = X_transf_df.values

    return X_array, Y_array, pipeline

def build_model():
    '''
    We use RandomForestClassifier for our machine learning algo.
    '''
    classifier = MultiOutputClassifier(estimator=RandomForestClassifier())

    return classifier

def evaluate_model(model, X_test, Y_test, category_names):
    '''
        INPUT:
        model - model that we created
        X_test - test portion of messages
        Y_test - test portion of labels
        category_names - label names. we loop across the label names to determine the score of our model for each label using classification_report

    '''
    yhat = model.predict(X_test)

    for i, c in enumerate(category_names):
        print(c)
        print(classification_report(Y_test[:, i], yhat[:, i]))


def save_model(model,pipeline,   model_filepath, pipeline_filepath):
    '''
        INPUT:
        model - model we created and trained
        model_filepath - where we want to store our model
    '''

    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

    #Save the pipeline that we use to apply count vectorizer and tfidf
    filename = pipeline_filepath
    pickle.dump(pipeline, open(filename, 'wb'))



def main():
    '''
    sys.argv takes in 4 arguments - script file, database_filepath, model_filepath, pipeline_filepath

    '''
    if len(sys.argv) == 4:
        database_filepath, model_filepath, pipeline_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_array, Y_array, pipeline = augment_data(X,Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.5, random_state=1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print(model.score(X_train, Y_train))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model,pipeline, model_filepath, pipeline_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()