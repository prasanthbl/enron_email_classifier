# Enron  email classifier

## Objective

The objective of the work is to classify the email dataset.
The enron email dataset is available in the below link
    https://bailando.berkeley.edu/enron_email.html

The data needs to be preprocessed / cleaned and then build a model to categorize them into the following classes
    
    a. Company Business, Strategy, etc. (elaborate in Section 3 [Topics])
   
    b. Purely Personal
   
    c) Personal but in professional context (e.g., it was good working with you)
   
    d) Logistic Arrangements (meeting scheduling, technical support, etc)
   
    e) Employment arrangements (job seeking, hiring, recommendations, etc)
   
    f) Document editing/checking (collaboration)
   
Train and test the model and report the performance

## Alogrithm / Approach

1. Download the email files from the weblink : https://bailando.berkeley.edu/enron_email.html
   and place it in the repository. The downloaded files are present in the folder "enron_with_categories
       
2. As per the category the sub folders are created for the classification and present in "enron_classifier_data"
   
3. Read the data for training and testing by using the load_files class in datasets.
4. Following preporcessing steps are done on the files.
    1. All the words are made to lower case
    2. Used the NLTK tokenizer 
    3. Fixed the minimum document frequency 
    4. Fixed the n-gram range
    5. removed the stop words. 
       Note: Stop words are used from NLTK and few words are white listed 
       based on the dataset.
       
5. Trained the Linear SVM model for the classification 

6. Performance measures such as confusion matrix and f1 score are computed for the model.

## Results

### Confusion Matrix 

### F1 score