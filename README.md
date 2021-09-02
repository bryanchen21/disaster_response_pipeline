# Disaster Response Pipeline Project

### Table of Contents
1. Project Motivation
2. File Descriptions
3. Results
4. Author and Acknowledgements


### Project Motivation

1. We analyse messages that were sent during real disasters either via social media or disaster response organisations to see what are the needs/labels of each message. 
2. We then come up with a model that can predict and classify messages.
3. Steps are as follows:
    - Build ETL pipeline that processes message and category data and loads to SQL lite database.
    - Machine learning pipeline will then read from the database and then save a multi output supervised learning model.
    - Web app will extract data from this database to provide data visualisation and will use our model to classify new messages for 36 categories.

### File Descriptions

- The "data" folder contains the 2 csv files from Figure 8 containing disaster messages along with their categories. It also contains the script to clean and merge the 2 csv files to create a sqlite database file.  
- The "models" folder contains the script used to train and output our model.
- The "app" folder contains the html templates and python script to launch the web app. 
### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/message_classifier`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/message_classifier.db models/classifier.pkl models/pipeline.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://localhost:3001/

### Discussion
1. Because we are dealing with text data and we need to extract the features to do modeling, we first use CountVectorizer and TFIDF transformer to turn each message to a vector of numbers - essentially a score representing how many times a word appears in the message weighted by how common the word appears across all the messages. In each message, the more common a word is across all messages, the lower its score. 
2. Then, we use the Random Forest Classifier for our prediction. 
3. In this multi-label classification problem where we want to predict suitable labels based on messages, in our dataset, the labels are unevenly distributed.
4. Labels such as 'offer', 'security', 'clothing', 'missing_people', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'fire', 'cold' are used on around 500 or less messages out of over 26,000 messages. This reduces the effectiveness of our machine learning algorithm when training the model on these minority labels. 
5. To overcome this data imbalance, we use data augmentation by oversampling these minority labels. We leverage the technique called multi-label synthetic minority over-sampling (MLSMOTE).
6. In our case, GridSearchCV did not return a good model compared to the baseline Random Forest Classifier model with default parameter values, so the baseline model is used.
7. Special thanks to this [article](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87) for providing the code to run MLSMOTE. 

### Results

- Web app shows the distribution of the messages dataset according to genre and labels. 
- The interface asks users for a message and returns the appropriate label.



### Author and Acknowledgements

- **Bryan Chen** - bryanchen21@gmail.com
- Credits to Figure 8 for the data. 



