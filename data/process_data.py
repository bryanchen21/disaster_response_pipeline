# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        INPUT:
        messages_filepath - this is the csv file for messages data
        categories_filepath - this is the csv file for messages data

        OUTPUT:
        2 dataframes for messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories

def clean_data(messages, categories):
    '''
        INPUT:
        messages and categories dataframe

        OUTPUT:
        df - merged dataframe with 36 columns for 36 labels
    '''
    df = messages.merge(categories,"left",on="id")

    # create a dataframe of the 36 individual category columns
    new_categories_df = df.categories.str.split(";",expand=True)

    # select the first row of the categories dataframe
    row = new_categories_df.iloc[0, :].tolist()

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x.split("-", 1)[0] for x in row]

    # rename the columns of `categories`
    new_categories_df.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string(the 1 or 0).
    # For example, `related - 0` becomes `0`, `related - 1` becomes `1`.Convert the string to a numeric value.
    for column in new_categories_df:
        # set each value to be the last character of the string
        new_categories_df[column] = [x[1] for x in new_categories_df[column].str.split('-', 1).tolist()]

        # convert column from string to numeric
        new_categories_df[column] = new_categories_df[column].astype(int)

    # drop the original categories column from `df`

    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, new_categories_df], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
        INPUT:
        df - merged dataframe from clean_data function

        OUTPUT:
        database_filename - name of database file
    '''
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('messages_categorised', engine, index=False)

    return print(pd.read_sql('SELECT * FROM messages_categorised limit 5', engine))
def main():

    "sys.argv takes in 4 arguments - script file, messages_filepath, categories_filepath, database_filepath"
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories= load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)

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