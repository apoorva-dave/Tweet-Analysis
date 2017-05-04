import string
import pandas as pd
from decorator import getfullargspec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
import random
import re
import numpy as np
import matplotlib.pyplot as plt
# Read data
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import dill
from sklearn.externals import joblib

def read_data():
    data = pd.read_csv("C:/Users/Apoorva/PycharmProjects/TweetsClassification/amazon-fine-foods/Reviews.csv", header=0, delimiter=",", quoting=15)

    print (data.shape)  # (25000, 3)
    #print (data["review"][0])  # Check out the review
    #print (data["sentiment"][0])  # Check out the sentiment (0/1)
    return data

# Shuffle the data

def randomize(data):
    # if data["Score"] < 3 :
    #     data["Score"] = 0
    # else:
    #     data["Score"] = 1
    data["Sentiment"] = data["Score"].apply(lambda score: "1" if score > 3 else "0")
    sentiment_data = list(zip(data["Text"], data["Sentiment"]))
    # random.shuffle(sentiment_data)
    return sentiment_data

# Generate training data

def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:13500])
    return list(train_X),list(train_y)

# Generate test data

def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[13501:25000])
    return list(test_X),list(test_y)

# Clean data.

def clean_text(text):
    text = text.replace("<br />", " ")
    # text = re.sub('[^A-Za-z0-9 ]+', '', str(text))
    text = text.encode().decode("utf-8")
    # Removing urls from text
    text = re.sub(r'http\S+', '', text,flags=re.MULTILINE)

    return text




def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

# Get Features matrix
def get_tweets_feature_matrix(classifier, data):
    vocab = classifier.named_steps['vectorizer'].get_feature_names()
    print(vocab)
    feature_matrix = []
    for line in data:
        tokenized_line = re.sub('[^A-Za-z0-9 ]+', '', str(line)).split(" ")
        line_matrix = [0] * len(vocab)
        for word in tokenized_line:
            if word in vocab:
                index_found = vocab.index(word)
                line_matrix[index_found] = line_matrix[index_found] + 1
            print(line_matrix)
        feature_matrix.append(line_matrix)

    return feature_matrix

def build_classifiers_unigrams(train_X,train_y):
        clf_logistic = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           # ! Comment line to include mark_negation and uncomment next line
                                           # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                           preprocessor=lambda text: text.replace("<br />", " "),
                                           max_features=10000)),
            ('classifier', LogisticRegression(C=1e5))
        ])
        clf_multinomialNB = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           # ! Comment line to include mark_negation and uncomment next line
                                           # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                           preprocessor=lambda text: text.replace("<br />", " "),
                                           max_features=10000)),
            ('classifier', MultinomialNB())
        ])

        clf_svm = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           # ! Comment line to include mark_negation and uncomment next line
                                           # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                           preprocessor=lambda text: text.replace("<br />", " "),
                                           max_features=10000)),
            ('classifier', LinearSVC())
        ])
        clf_bernoulliNB = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           # ! Comment line to include mark_negation and uncomment next line
                                           # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                           preprocessor=lambda text: text.replace("<br />", " "),
                                           max_features=10000)),
            ('classifier', BernoulliNB())
        ])
        clf = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                            stop_words=stopwords.words('english') + list(string.punctuation))),
             ('classifier', MultinomialNB(alpha=0.05)),
         ])

        model_logistic = clf_logistic.fit(train_X, train_y)
        filename = 'model_logistic.sav'
        with open(filename, 'wb') as f:
             dill.dump(model_logistic,f)

        model_multinomial = clf_multinomialNB.fit(train_X, train_y)
        filename = 'model_multinomial.sav'
        with open(filename, 'wb') as f:
             dill.dump(model_multinomial,f)

        # SVM MODEL
        filename = 'model_svm.sav'

        model_svm = clf_svm.fit(train_X, train_y)
        with open(filename, 'wb') as f:
            dill.dump(model_svm,f)

        filename = 'model_bernoulli.sav'
        model_bernoulli = clf_bernoulliNB.fit(train_X, train_y)
        with open(filename, 'wb') as f:
            dill.dump(model_bernoulli, f)

def formatt(x):
        if x == '0':
            return 0
        return 1

def unigrams(test_X,test_y,data):

    prediction = dict()
    print("Loading..")
    filename = 'model_multinomial.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)
    print("Loading..")

    prediction['Multinomial'] = clf2.predict(test_X)
    # prediction['Multinomial'] = model_multinomial.predict(test_X)
    print("Predicting Multinomial NB..")
    print(prediction['Multinomial'])
    print("Printing Multinomial NB accuracy")       #0.843
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_multinomialNB.score(test_X, test_y))


    # SVM MODEL
    filename = 'model_svm.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['svm'] = clf2.predict(test_X)
    print("Predicting SVM..")
    print(prediction['svm'])
    print("Printing SVM accuracy")                  #0.824
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_svm.score(test_X, test_y))


    #BERNOULLI NB

    filename = 'model_bernoulli.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Bernoulli'] = clf2.predict(test_X)
    print("Predicting Bernoulli NB..")
    print(prediction['Bernoulli'])
    print("Printing Bernoulli NB accuracy")                #0.77
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)
    print(result)

    #Logistic Regression

    filename = 'model_logistic.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Logistic'] = clf2.predict(test_X)
    print("Predicting Logistic Regression..")
    print(prediction['Logistic'])
    print("Printing Logistic Regression accuracy")
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)             #0.821
    print(result)

    print("Confusion matrix for Multinomial ..")
    print(metrics.classification_report(test_y, prediction['Multinomial'], target_names=["positive", "negative"]))

    print("Confusion matrix for Logistic Regression ..")
    print(metrics.classification_report(test_y, prediction['Logistic'], target_names=["positive", "negative"]))

    print("Confusion matrix for SVM ..")
    print(metrics.classification_report(test_y, prediction['svm'], target_names=["positive", "negative"]))

    print("Confusion matrix for Bernoulli ..")
    print(metrics.classification_report(test_y, prediction['Bernoulli'], target_names=["positive", "negative"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k']
    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        myarray = np.array(test_y)
        # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = int(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #feature_matrix = get_tweets_feature_matrix(clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)
    # print("Getting feature names")
    # print(clf.named_steps['vectorizer'].get_feature_names())

def build_classifiers_bigram(train_X,train_y):

    print("Building models..")

    bigram_clf_multinomial = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', MultinomialNB())
    ])
    bigram_clf_logistic = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LogisticRegression(C=1e5))
    ])
    bigram_clf_svm = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LinearSVC())
    ])
    bigram_clf_bernoulli = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', BernoulliNB())
    ])
    #Logistic

    bigram_model_logistic = bigram_clf_logistic.fit(train_X, train_y)
    filename = 'bigram_model_logistic.sav'
    with open(filename, 'wb') as f:
        dill.dump(bigram_model_logistic, f)

    #Multinomial

    bigram_model_multinomial = bigram_clf_multinomial.fit(train_X, train_y)
    filename = 'bigram_model_multinomial.sav'
    with open(filename, 'wb') as f:
        dill.dump(bigram_model_multinomial, f)

    # SVM MODEL

    filename = 'bigram_model_svm.sav'
    bigram_model_svm = bigram_clf_svm.fit(train_X, train_y)
    with open(filename, 'wb') as f:
        dill.dump(bigram_model_svm, f)

    filename = 'bigram_model_bernoulli.sav'
    bigram_model_bernoulli = bigram_clf_bernoulli.fit(train_X, train_y)
    with open(filename, 'wb') as f:
        dill.dump(bigram_model_bernoulli, f)

    # bigram_clf_bernoulli.fit(train_X, train_y)


    # print("Printing accuracy")
    # print(bigram_clf.score(test_X, test_y))



def bigrams(test_X,test_y,data):

    prediction = dict()
    print("Loading..")
    filename = 'bigram_model_multinomial.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)
    print("Loading..")

    prediction['Bigram_Multinomial'] = clf2.predict(test_X)
    # prediction['Multinomial'] = model_multinomial.predict(test_X)
    print("Predicting Multinomial NB for bigrams..")
    print(prediction['Bigram_Multinomial'])
    print("Printing Multinomial NB accuracy for bigrams")               #0.815
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_multinomialNB.score(test_X, test_y))


    # SVM MODEL
    filename = 'bigram_model_svm.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Bigram_svm'] = clf2.predict(test_X)
    print("Predicting SVM for bigrams..")
    print(prediction['Bigram_svm'])
    print("Printing SVM accuracy for bigrams")                  #0.86
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_svm.score(test_X, test_y))


    #BERNOULLI NB

    filename = 'bigram_model_bernoulli.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Bigram_Bernoulli'] = clf2.predict(test_X)
    print("Predicting Bernoulli NB for bigrams..")
    print(prediction['Bigram_Bernoulli'])
    print("Printing Bernoulli NB accuracy for bigrams")             #0.769
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)
    print(result)

    #Logistic Regression

    filename = 'bigram_model_logistic.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Bigram_Logistic'] = clf2.predict(test_X)
    print("Predicting Logistic Regression for bigrams..")
    print(prediction['Bigram_Logistic'])
    print("Printing Logistic Regression accuracy for bigrams")
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)             #0.86
    print(result)

    print("Confusion matrix for Multinomial Bigram..")
    print(metrics.classification_report(test_y, prediction['Bigram_Multinomial'], target_names=["positive", "negative"]))

    print("Confusion matrix for Logistic Regression Bigram..")
    print(metrics.classification_report(test_y, prediction['Bigram_Logistic'], target_names=["positive", "negative"]))

    print("Confusion matrix for SVM Bigram ..")
    print(metrics.classification_report(test_y, prediction['Bigram_svm'], target_names=["positive", "negative"]))

    print("Confusion matrix for Bernoulli Bigram..")
    print(metrics.classification_report(test_y, prediction['Bigram_Bernoulli'], target_names=["positive", "negative"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k']
    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        myarray = np.array(test_y)
        # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = int(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #feature_matrix = get_tweets_feature_matrix(clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)
    # print("Getting feature names")
    # print(clf.named_steps['vectorizer'].get_feature_names())








    # feature_matrix = get_tweets_feature_matrix(bigram_clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)

    #print("Getting feature names")
    #print(bigram_clf.named_steps['vectorizer'].get_feature_names())

def build_classifiers_unigram_bigram(train_X,train_y):

    print("Building models..")

    unigram_bigram_clf_multinomial = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', MultinomialNB())
    ])
    unigram_bigram_clf_logistic = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LogisticRegression(C=1e5))
    ])
    unigram_bigram_clf_svm = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LinearSVC())
    ])
    unigram_bigram_clf_bernoulli = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', BernoulliNB())
    ])
    #Logistic

    unigram_bigram_model_logistic = unigram_bigram_clf_logistic.fit(train_X, train_y)
    filename = 'unigram_bigram_model_logistic.sav'
    with open(filename, 'wb') as f:
        dill.dump(unigram_bigram_model_logistic, f)

    #Multinomial

    unigram_bigram_model_multinomial = unigram_bigram_clf_multinomial.fit(train_X, train_y)
    filename = 'unigram_bigram_model_multinomial.sav'
    with open(filename, 'wb') as f:
        dill.dump(unigram_bigram_model_multinomial, f)

    # SVM MODEL

    filename = 'unigram_bigram_model_svm.sav'
    unigram_bigram_model_svm = unigram_bigram_clf_svm.fit(train_X, train_y)
    with open(filename, 'wb') as f:
        dill.dump(unigram_bigram_model_svm, f)

    filename = 'unigram_bigram_model_bernoulli.sav'
    unigram_bigram_model_bernoulli = unigram_bigram_clf_bernoulli.fit(train_X, train_y)
    with open(filename, 'wb') as f:
        dill.dump(unigram_bigram_model_bernoulli, f)

    # bigram_clf_bernoulli.fit(train_X, train_y)


    # print("Printing accuracy")
    # print(bigram_clf.score(test_X, test_y))


def unigram_bigram(test_X,test_y,data):

    prediction = dict()
    print("Loading..")
    filename = 'unigram_bigram_model_multinomial.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)
    print("Loading..")

    prediction['Unigram_Bigram_Multinomial'] = clf2.predict(test_X)
    # prediction['Multinomial'] = model_multinomial.predict(test_X)
    print("Predicting Multinomial NB for unigrams bigrams..")
    print(prediction['Unigram_Bigram_Multinomial'])
    print("Printing Multinomial NB accuracy for unigrams bigrams")               #0.81
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_multinomialNB.score(test_X, test_y))


    # SVM MODEL
    filename = 'unigram_bigram_model_svm.sav'

    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Unigram_Bigram_svm'] = clf2.predict(test_X)
    print("Predicting SVM for unigrams bigrams..")
    print(prediction['Unigram_Bigram_svm'])
    print("Printing SVM accuracy for unigrams bigrams")                  #0.86
    result = clf2.score(test_X, test_y)
    print(result)

    # print(clf_svm.score(test_X, test_y))


    #BERNOULLI NB

    filename = 'unigram_bigram_model_bernoulli.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Unigram_Bigram_Bernoulli'] = clf2.predict(test_X)
    print("Predicting Bernoulli NB for unigrams bigrams..")
    print(prediction['Unigram_Bigram_Bernoulli'])
    print("Printing Bernoulli NB accuracy for unigrams bigrams")             #0.771
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)
    print(result)

    #Logistic Regression

    filename = 'unigram_bigram_model_logistic.sav'
    with open(filename, 'rb') as f:
        clf2 = dill.load(f)

    prediction['Unigram_Bigram_Logistic'] = clf2.predict(test_X)
    print("Predicting Logistic Regression for unigrams bigrams..")
    print(prediction['Unigram_Bigram_Logistic'])
    print("Printing Logistic Regression accuracy for unigrams bigrams")         #0.86
    # print(clf_bernoulliNB.score(test_X, test_y))

    result = clf2.score(test_X, test_y)             #0.86
    print(result)

    print("Confusion matrix for Multinomial Unigram Bigram..")
    print(metrics.classification_report(test_y, prediction['Unigram_Bigram_Multinomial'], target_names=["positive", "negative"]))

    print("Confusion matrix for Logistic Regression Unigram Bigram..")
    print(metrics.classification_report(test_y, prediction['Unigram_Bigram_Logistic'], target_names=["positive", "negative"]))

    print("Confusion matrix for SVM Unigram Bigram ..")
    print(metrics.classification_report(test_y, prediction['Unigram_Bigram_svm'], target_names=["positive", "negative"]))

    print("Confusion matrix for Bernoulli Unigram Bigram..")
    print(metrics.classification_report(test_y, prediction['Unigram_Bigram_Bernoulli'], target_names=["positive", "negative"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k']
    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        myarray = np.array(test_y)
        # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = int(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #feature_matrix = get_tweets_feature_matrix(clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)
    # print("Getting feature names")
    # print(clf.named_steps['vectorizer'].get_feature_names())


    # feature_matrix = get_tweets_feature_matrix(bigram_clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)

    #print("Getting feature names")
    #print(bigram_clf.named_steps['vectorizer'].get_feature_names())





    # feature_matrix = get_tweets_feature_matrix(unigram_bigram_clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)

    # test_data = ["I am happy with your airlines","I hate you"]
    # predicted = unigram_bigram_clf.predict(test_data)
    # print(predicted)

    # Check the feature names
    #print("Getting feature names")
    #print (unigram_bigram_clf.named_steps['vectorizer'].get_feature_names())


def main():
    print("Reading Data..")
    data = read_data()
    print("Data Fetched")
    # print(data["review"][0])
    sentiment_data = randomize(data)
    print("Dividing into train data..")
    train_X, train_y = train_data(sentiment_data)
    print("Dividing into test data..")
    test_X,test_y = test_data(sentiment_data)
    print("Cleaning data..")
    for i in range(0, len(train_X)):
        train_X[i] = clean_text(str(train_X[i]))
    # # print(train_X[0])
    # # print(train_y[0])
    #
    for i in range(0, len(test_X)):
        test_X[i] = clean_text(str(test_X[i]))
    # print(test_X[0])
    # print(test_y)
    # exit(0)
    #print(random_data)

    #Call this function to create and save models in file

    # build_classifiers_unigrams(train_X,train_y)
    # build_classifiers_bigram(train_X,train_y)

    # Load saved models to test for accuracy
    # unigrams(test_X,test_y,data)
    # bigrams(test_X,test_y,data)
    build_classifiers_unigram_bigram(train_X,train_y)
    unigram_bigram(test_X,test_y,data)  #0.81



main()

