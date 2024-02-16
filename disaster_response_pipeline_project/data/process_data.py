import sys
import pandas as pd
from sqlalchemy import create_engine

# database_filepath = sys.argv[1:]
# database_filepath = 'DisasterResponse.db'


def load_data_cat(categories_filepath):
    """Load category data from a CSV file."""
    categories = pd.read_csv(categories_filepath)
    return categories

def load_data_mes(messages_filepath):
    """Load message data from a CSV file."""
    messages = pd.read_csv(messages_filepath)
    return messages

def clean_data(df, categories):
    """Clean the data by combining message and category dataframes."""
    # Merge message and category dataframes
    df = pd.merge(df, categories, on='id')
    
    # Split the category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    
    # Rename the category columns
    categories.columns = category_colnames
    
    # Convert category values to binary numbers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
    
    # Replace the 'categories' column with the categories dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, db_filename):
    engine = create_engine(f'sqlite:///{db_filename}')
    
    # Save the clean dataset into the SQLite database
    df.to_sql('samtable', engine, index=False, if_exists='replace')
    
    # Print a message indicating the successful save
    print("Clean dataset saved to SQLite database.")

def main():
    """Main entry point."""
    if len(sys.argv) == 4:
        mes_file_path = '/Users/samuelmilazzo/Desktop/samuel 2023/learning/project2/disaster_response_pipeline_project/data/disaster_messages.csv'
        cat_file_path = '/Users/samuelmilazzo/Desktop/samuel 2023/learning/project2/disaster_response_pipeline_project/data/disaster_categories.csv'
        database_filepath = 'DisasterResponse.db'
        
        print('Loading data...')
        messages = load_data_mes(mes_file_path)
        categories = load_data_cat(cat_file_path)
        
        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...')
        save_data(df=df,db_filename=database_filepath)
        
        print('Cleaned data saved to the SQLite database!')
    else:
        print('Please provide the filepaths of the messages, categories, and SQLite database '
              'as arguments in the command line.')

if __name__ == '__main__':
    main()
