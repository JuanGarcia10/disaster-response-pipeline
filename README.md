# Introduction

This project is part of the Udacity Nanodegree "Data Scientist" and illustrates the creation of a machine learning pipeline

# Disaster Response

During a (natural) disaster thousands of posts on social media and the news will be posted. It is critical for disaster-response-teams to be able to analyze and identify the relevant messages to take action. Just by searching for a word is not enough to make a meaningful categorization. Here is where machine learning comes into play!

## Data Set

The data set was provided by the folks from [appen](https://appen.com/), previously known as "figure eight". It contains real text messages already classified into meaningful categories.

## Methodology

### Data Cleaning

Data Cleaning activities are summarized in the Jupyter Notebook "ETL Pipeline Preparation". By the end of the notebook a SQL database is created with data ready to be analyzed. I am prividing the SQL database but not the raw data.

### Feature Extraction and Data Modeling

We are analyzing only strings (text) and that is why we use NLP (Natural language processing) machine learning methods for the feature extraction. Please refer to the Jupyter Notebook "ML Pipeline Preparation" for further information.

I use grid search to find the best combination of parameters for my model. By the end of this Notebook a Model based on a Random Forest Classifier will be saved under the name "classifier.pkl".

For the final script (train_classifier) I use a random forest classifier and display the most important metrics such as accuracy, precision and the f1-score. 

### Web-App and Visualization

Using a web-app a firs-response team can quickly categorize any text message and decide wheter to take action or not. Furthermore, you will find some plots that give a basic overview of the training data.

## What do you need to know to run this project by yourself?

### Libraries

You need Python 3.6 or above as well as the following libraries
- Numpy
- Pandas
- NLKT
- SQLAlchemy
- Scikit-Learn
- Joblib
- Flask
- Plotly

Please refer to the Jupiter Notebooks to find the concrete modules of the libraries

### Procedure

- Clean the data ("ETL Pipeline Preparation")
- Fit the model ("ML Pipeline Preparation") and save the model under the name "classifier.pkl"
- Create pipeline: Use Jupiter Notebooks to create python scripts:  
  - process_data.py (data cleaning)
  - train_classifier.py (fitting the model)
- Create a web-app:
  - create templates for web-app (folder templates)
  - create python script to run the app (run.py)

to start the we application, go to the project's directory and run:
```
python run.py
```
once the script is running go to your favorite browser (hopefully Mozilla) and go to `http://0.0.0.0.3001`. If this does not work please try `http//:localhos:3001`.

Enjoy!
