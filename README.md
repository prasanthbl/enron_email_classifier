# Enron  email classifier

## Objective

The objective of the work is to classify the email dataset available in the below link
    https://bailando.berkeley.edu/enron_email.html

The data needs to be preprocessed / cleaned and then build a classier model to categorize them into the following classes. Then train and test the model and report the performance.
    
    a) Company Business, Strategy, etc.
   
    b) Purely Personal
   
    c) Personal but in professional context (e.g., it was good working with you)
   
    d) Logistic Arrangements (meeting scheduling, technical support, etc)
   
    e) Employment arrangements (job seeking, hiring, recommendations, etc)
   
    f) Document editing/checking (collaboration)
   


## Alogrithm / Approach

### Enron email Dataset
1. Download the email files from the weblink : https://bailando.berkeley.edu/enron_email.html and place it in the repository. The downloaded files are present in the folder "enron_with_categories"
     
2. The *.cats file have the category of the email data.
3. There are total 1704 emails 
4. As per the category there are 8 sub folders present in the downloaded data set. Category 7 and Category 8 belongs to empty email dataset and it is not considered in this assignment.
5. There are multiple category assigned to the same email data in the *.cats file. Thus to avoid redundancy, I just kept the email in one of the category. The category wise size of the data set is shown below, with a total of 1663 emails.
   
   class| No of email
   ---| --- |
   class  1 | 855
   class 2 | 48
   class 3 | 135
   class 4 | 426
   class 5 | 64
   class 6 | 135
   
6. From the above point it is very clear that the dataset is not a balanced dataset. Class 2 have a data size of 48. Thus I have down sampled the dataset for class having size higher than 48.

7. Thus now each class have a data set with size 48, with a total of 288 which is used in training and testing email classifier.
   
### Data pre-processing 
1. Read the data by using the load_files class in datasets
2. For ease of use the dataset is conveted into dataframes using pandas library. 
3. Following preporcessing steps are done on the email data.

    1. Used the NLTK tokenizer, to tokenize the email data. 
    2. Consider the tokens with only alphabets and other token having numeric,special characters are filtered.
    3. All the tokens are made to lower case as the case of the word is not conveying any special meaning to us. 
    4. The stop words from the NLTK library is extended with, custom stop words that are particularly applicable for this corpus. These custom stop words are identified by visual inspection of the email data files. There are still further improvements can be done making the scripts that will extract only the content of the body from the email data. Regular expressoion based processing also could be done to extract desired text from the email data.
    4. Removed the tokens with a length of 1 and 2 as they are not going to convey any significant meaning to us. 
    5. Removed uncommon words.
    6. Wordnet lemmatizer from the NLTK library is used.
    7. The tokens are again combined as a string for feature extraction.
    8. freq_word function is used to identify the stop words. There are still improvements we can do by some way of removing the names from the email data.
       
### Feature extraction

1. countvectorizer from NLTK library is used for. With the following parameters.

    1. Fixed the n-gram range as 1.
    2. Removed the stop words. 
    3. Minimum document frequency is set as 2.
    
2. tfidf feature is extracted for the email dataset and it is used for training the Multinominal Naive Bayes / SVM classifier.

### Classifier design
1. Trained the Multinominal Naive Bayes and Linear SVM model for the classification.
2. The dataset is splitted into 80% training and 20% for validation. 
3. Hyperparameters of both the models are tuned for achieving higher performance. It can be found using grid search and cross validation. 
4. Performance measures such as confusion matrix and the accuracy are computed for the model.

## Results
1. Multinominal Naive Bayes classifier gives 62% accuracy on test data.
   
   ![Alt text](images/mn_results.png?raw=true "MNB")
2. SVM classifier gives 67.2% accuracy on the test data.
   ![Alt text](images/svm_results.png?raw=true "SVM")
   
   Note: In the confution matrix index represents as below

    0 ->  Company Business, Strategy, etc.
   
    1 -> Purely Personal 
   
    2 -> Personal but in professional context (e.g., it was good working with you)
   
    3 -> Logistic Arrangements (meeting scheduling, technical support, etc)
   
    4 -> Employment arrangements (job seeking, hiring, recommendations, etc)
   
    5 -> Document editing/checking (collaboration)




