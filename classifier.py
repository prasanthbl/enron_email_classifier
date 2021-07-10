import numpy as np
import pandas as pd
from sklearn import (
    datasets, feature_extraction,svm
)
import collections
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Reading the email dataset using load files the entire dataset present in the folder is
# read and stored in the variable email_data
x, y = [], []
email_data = datasets.load_files(
    'test', shuffle=True, random_state=42, encoding="utf-8")
X = np.append(x, email_data.data)
y = np.append(y, email_data.target)


# Stored the data to a CSV file to visualize the content and reference data before processing.
data = {}
data['Email'] = X
data['Type'] = y
data = pd.DataFrame(data)
data.to_csv("Enron_Email_Dataset.csv", index = False)


# Using the stopwords and lemmatizer from NLTK
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()
# Analyzed the email data and created a set of ord
custom_stop_words = ['message-id', 'date', 'from', 'to', 'subject', 'mime-version', 'content-type',
                      'content-transfer-encoding', 'x-from', 'x-to', 'x-cc', 'x-bcc', 'x-folder',
                      'x-origin', 'x-filename', 'enron', 'com','7bit', 'thyme', 'pst', 'mon', 'tue',
                     'wed','thu','fri','sat','sun','jan','feb','mar','apr','may','jun','jul',
                     'aug','sep','oct','nov','dec','pdt','http','hi','cc', 'bcc','pm','ee','kean', 'vince','urszula',
                     'steven', 'message', 'sent','kaminski', 'vkamins','mailto','sobczyk','piotr','houston'
                     'rkean', 'skean','forwarded','melissa','lieberman','vkaminski','houston','lipca','warszawy', 'powrot',
                     'cena','czy','kontakt','phil']

stop_words.extend(custom_stop_words)

# Data pre processing
X_treated = []
for i in range(0, data.shape[0]):
    # Tokenization
    processed_tokens = nltk.word_tokenize(str(data['Email'][i]))
    # Filtering the token containing only alphabets
    processed_tokens = [w for w in processed_tokens if w.isalpha()]
    # Converting all the tokens to lower case
    processed_tokens = [w.lower() for w in processed_tokens]
    # remove the stop words from NLTK library and also remove the custom stop words
    processed_tokens = [w for w in processed_tokens if w not in stop_words]
    # remove the uncommon words
    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    # remove words with length 1 and 2
    processed_tokens = [w for w in processed_tokens if len(w) != (1)]
    processed_tokens = [w for w in processed_tokens if len(w) != (2)]
    # lemmatize
    processed_tokens = [lemmatizer.lemmatize(w) for w in processed_tokens]
    # combining all the tokens into a single s
    processed_tokens = ' '.join(processed_tokens)
    X_treated.append(processed_tokens)

data['treated_mails'] = X_treated

# # Words which are meaning less and doesn't have any significance is obtained from this function
# # and manually added to the custom_stop_words.
# def freq_words(x, terms = 10):
#     all_words = ' '.join([text for text in x])
#     all_words = all_words.split()
#
#     fdist = nltk.FreqDist(all_words)
#     words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
#
#     # selecting top most frequent words
#     d = words_df.nlargest(columns="count", n = terms)
#
# freq_words(data[data['Type'] == (1 or 2 or 3 or 0 or 4 or 5 or 6)]['treated_mails'], 50)


# Feature extraction
# setting the parameter for countvectorizer
count_vectorizer = feature_extraction.text.CountVectorizer(
    lowercase=True,  # for demonstration, True by default
    tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
    min_df=2,  # minimum document frequency, i.e. the word must appear more than once.
    ngram_range=(1, 1),
    stop_words=custom_stop_words # again removing the custom_stop
)

# TFIDF feature is extracted on the processed data
X_tfidf = count_vectorizer.fit_transform(data['treated_mails']).toarray()
X_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(
        X_tfidf).toarray()
y = data['Type'].values


# Training the classifier
# Split the data set into training and testing dataset. 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state = 8)

# Creating the Multinominal Naive Bayes model
Naive_model = MultinomialNB(alpha=0.1).fit(X_train, y_train)
# Evaluating the performance of training dataset
predictions_training = Naive_model.predict(X_train)
score = accuracy_score(y_train,predictions_training)
print(f"MultinominalNB training data Accuracy: {score*100:.2f}%")

# confusion matrix for the training dataset
print("Training dataset confusion matrix:")
y_true = pd.Series(y_train, name='Real')
y_pred = pd.Series(predictions_training, name='Predicted')
print(pd.crosstab(y_true, y_pred))

# Evaluating the performance of testing dataset
predictions_test = Naive_model.predict(X_test)
score = accuracy_score(y_test, predictions_test)
print(f"MultinominalNB test data Accuracy: {score*100:.2f}%")

# confusion matrix for the testing dataset
print("Test dataset confusion matrix:")
y_true = pd.Series(y_test, name='Real')
y_pred = pd.Series(predictions_test, name='Predicted')
print(pd.crosstab(y_true, y_pred))

# Creating the SVM model
SVM = svm.SVC(C=0.8, kernel='linear', gamma='auto', decision_function_shape='ovo')
SVM.fit(X_train, y_train)
# Evaluating the performance of training dataset
predictions_SVM = SVM.predict(X_train)
score = accuracy_score(y_train,predictions_SVM)
# Use accuracy_score function to get the accuracy
print(f"SVM training data accuracy : {score*100:.2f}%")

# confusion matrix for the training dataset
print("Training dataset confusion matrix:")
y_true = pd.Series(y_train, name='Real')
y_pred = pd.Series(predictions_SVM, name='Predicted')
print(pd.crosstab(y_true, y_pred))

# Evaluating the performance of testing dataset
predictions_test = SVM.predict(X_test)
score = accuracy_score(y_test, predictions_test)
# Use accuracy_score function to get the accuracy
print(f"SVM test data accuracy : {score*100:.2f}%")

# confusion matrix for the test dataset
print("Test set confusion matrix:")
y_true = pd.Series(y_test, name='Real')
y_pred = pd.Series(predictions_test, name='Predicted')
print(pd.crosstab(y_true, y_pred))