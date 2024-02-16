# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# project2

## Library Used 

- Python 3
- Pandas
- sqlalchemy
- Numpy 
- sqlalchemy 
- sklearn
- pickle

## File Description 

- `ETL Pipeline Preparation.ipynb`: Jupyter Notebook containing the analysis of ETL pipeline
- `ML Pipeline Preparation.ipynb`: Jupyter Notebook containing the analysis of ML pipeline

## Index

- Project2
    - ETL pipeline
        - import all the relative libraries 
        - load messages.csv into a dataframe
        - load the categories.csv into a dataframe
        - merge datasets
        - create a dataframe of the 36 individual category columns
        - select the first row of the categories dataframe
        - convert the values to strings and use a lambda function to extract new column names
        - rename the columns of `categories`
        - create an iteration to convert category in 0 or 1 number
        - replace 'categories' column in df with new category columns
        - concatenate the original dataframe with the new 'categories' dataframe
        - check how many duplicates are in the dataset
        - drop the duplicates
        - create an SQLite engine
        - save the clean dataset into the SQLite database
    - check all the details in ML pipeline
        - import all the relative libraries 

## Link to Git : https://github.com/samusam91/project2

None

## Acknowledgement

None
