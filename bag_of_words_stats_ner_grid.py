
# coding: utf-8

# In[36]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import time
import spacy
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')
import utils


# In[3]:

nlp = spacy.load("en_core_web_md")


# In[4]:

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')


# In[5]:

train.head(2)


# In[6]:

test.head(2)


# In[7]:

print("Number of labelled training examples: {}".format(train.shape[0]))


# In[8]:

print("Distribution of training data for the three authors")
train['author'].value_counts().plot(kind="barh", color='brg')


# In[9]:

print("Number of unlabelled test examples: {}".format(test.shape[0]))


# In[10]:

# simple split for screening model performance
X_train, X_test, y_train, y_test = train_test_split(train, train.author, test_size=0.20, random_state=42)


# In[11]:

# n-fold stratified CV for robust model performance
X = train
y = train.author


# In[12]:

le = preprocessing.LabelEncoder()
le.fit(y)


# In[13]:

le.classes_


# In[14]:

# sanity check for label ordering
# 0: EAP, 1: HPL, 2: MWS


# In[15]:

y[:5]


# In[16]:

le.transform(y)[:5]


# ## Author-specific Named Entities

# In[17]:

ner_EAP = pd.read_csv('data/ner_EAP.csv', header=None)
ner_EAP = ner_EAP[0].tolist()
ner_HPL = pd.read_csv('data/ner_HPL.csv', header=None)
ner_HPL = ner_HPL[0].tolist()
ner_MWS = pd.read_csv('data/ner_MWS.csv', header=None)
ner_MWS = ner_MWS[0].tolist()


# In[18]:

ner_EAP[:5]


# In[19]:

def kfold_CV(clf, X, y, folds, transform=True):
    """ Run a stratified k-fold Cross Validation on the training set and print the results.
        
        Args:
            clf (Pipeline): sklearn Pipeline
            X    (pandas df): data points, here: novel snippets
            y    (pandas df): class labels, here: authors
            folds      (int): number of folds
            transform (bool): if True, use .fit_transform(); if False, use .fit()
    """

    kf = StratifiedKFold(n_splits=folds, shuffle=True)
    
    precision, recall, f1 = [], [], []

    fold_cntr = 1
    for train_index, test_index in kf.split(X,y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if transform == True:
            clf.fit_transform(X_train, y_train)
        else:
            clf.fit(X_train, y_train)
        
        predicted = text_clf.predict(X_test)
        prec_, rec_, f1_ = precision_recall_fscore_support(y_test, predicted, average='macro')[:3]
        
        precision.append(prec_)
        recall.append(rec_)
        f1.append(f1_)
        
        print("FOLD: {} Precision: {}, Recall: {}, F1: {}".format(fold_cntr, round(prec_,3), round(rec_,3), round(f1_,3)))
        fold_cntr += 1
        
    print("\nAverage results of {}-fold stratified CV\n".format(folds))
    print("Precision: {}, Standard deviation: {}".format(np.mean(precision), np.std(precision)))
    print("Recall:    {}, Standard deviation: {}".format(np.mean(recall), np.std(recall)))
    print("Macro f1:  {}, Standard deviation: {}".format(np.mean(f1), np.std(f1)))


# In[46]:

class DataFrameColumnExtracter(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


# In[21]:

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # print([{'length': len(text), 'num_sentences': text.count('.')} for text in posts])
        return [{'length': len(text), 'num_sentences': text.count('.')} for text in posts]


# In[22]:

class NER_extractor(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        """The workhorse of this feature extractor"""
        ner_list = []
        for text in posts:
            doc = nlp(text)
            eap_ners = 0
            hpl_ners = 0
            mws_ners = 0
            if doc.ents:
                for ent in doc.ents:
                    if ent.text in ner_EAP:
                        eap_ners += 1
                    elif ent.text in ner_HPL:
                        hpl_ners += 1
                    elif ent.text in ner_MWS:
                        mws_ners += 1
                    
            # print({'eap_ners':eap_ners, 'hpl_ners':hpl_ners, 'mws_ner':mws_ners})
            ner_list.append({'eap_ners':eap_ners, 'hpl_ners':hpl_ners, 'mws_ner':mws_ners})
        return ner_list


# In[76]:

# Use FeatureUnion to combine the features
featureunionvect = FeatureUnion(
    transformer_list=[

        # Pipeline for pulling features from the text snippet
        ('text_snippet', Pipeline([
            ('selector', DataFrameColumnExtracter('text')),
            ('vec', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),

        # Pipeline for pulling ad hoc features text snippet
        ('text_stats', Pipeline([
            ('selector', DataFrameColumnExtracter('text')),
            ('stats', TextStats()),  # returns a list of dicts
            ('stats_vec', DictVectorizer()),  # list of dicts -> feature matrix
        ])),

        # Pipeline for pulling NER features text snippet
        ('text_ner', Pipeline([
            ('selector', DataFrameColumnExtracter('text')),
            ('ner', NER_extractor()),  # returns a list of dicts
            ('ner_vect', DictVectorizer()),  # list of dicts -> feature matrix
        ])),

    ],

    # weight components in FeatureUnion
    transformer_weights=None,
)


# In[77]:

classifier = MultinomialNB()


# In[78]:

pipeline = Pipeline([('vect', featureunionvect), ('classifier', classifier)])


# In[95]:

X_train_sample = X_train.head(100)
y_train_sample = y_train.head(100)


# In[100]:

parameters = {
    'vect__text_snippet__vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),
    'vect__text_snippet__vec__max_features': (None, 5000, 10000, 20000),
    'vect__text_snippet__vec__min_df': (1, 5, 10),#, 20, 50), 
    'vect__text_snippet__tfidf__use_idf': (True, False),
    'vect__text_snippet__tfidf__sublinear_tf': (True, False),
    'vect__text_snippet__vec__binary': (True, False),
    'vect__text_snippet__tfidf__norm': ('l1', 'l2'),
    'classifier__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    }  


# In[101]:

grid_search = GridSearchCV(pipeline, parameters, verbose=2)


# In[ ]:

t0 = time.time()
grid_search.fit(X_train_sample, y_train_sample)
print("done in {0}s".format(time.time() - t0))
print("Best score: {0}".format(grid_search.best_score_))
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(list(parameters.keys())):
    print("\t{0}: {1}".format(param_name, best_parameters[param_name]))

