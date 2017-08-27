Basics in Text-Mining
======================

Tokenisation
-------------
Tokenising a document is a preprocessing stage that involves both data cleaning and feature extraction.
The eventual output is an extracted list of word or words.

Stopwords
**********
Stopwords are common words that are found in most texts in a corpus. 


Tokenizing
***********

.. code:: python
  
  import nltks
  
  # Word Tokens
  text11 = "Children shouldn't drink a sugary drink before bed."
  print text11.split(' ')
  ['Children', "shouldn't", 'drink', 'a', 'sugary', 'drink', 'before', 'bed.']

  print nltk.word_tokenize(text11)
  ['Children', 'should', "n't", 'drink', 'a', 'sugary', 'drink', 'before', 'bed', '.']

Sentences can be tokenised too.

.. code:: python

  import nltks

  # Sentence Tokens
  text12 = "This is the first sentence. A gallon of milk in the U.S. \
              costs $2.99. Is this the third sentence? Yes, it is!"
  print nltk.sent_tokenize(text12)
  ['This is the first sentence.',
   'A gallon of milk in the U.S. costs $2.99.',
   'Is this the third sentence?',
   'Yes, it is!']


Normalisation
**************

There are various ways of normalising text. 

**Changing Case**

An option is to change all text to lowercase or uppercase. Note that preservation of uppercase might be relevant in certain cases.

.. code:: python

  str = ['A', 'b', 'c', 'd']
  print [str.lower() for i in str]
  >>> ['a', 'b', 'c', 'd']

**Lemmatisation**

Lemmatisation will convert words to its root word. 
It is a variant of stemming, but the latter might return part of a word, e.g., sensing > sens.
Lemmatising will always give but a word.

.. code:: python

  udhr = nltk.corpus.udhr.words('English-Latin1')

  # Using Stemming
  porter = nltk.PorterStemmer()
  print [porter.stem(t) for t in udhr[:20]]
  [u'univers',
   u'declar',
   u'of',
   u'human',
   u'right',
   u'preambl',
   u'wherea',
   u'recognit']

  # Using Lemmatization
  WNlemma = nltk.WordNetLemmatizer()
  print [WNlemma.lemmatize(t) for t in udhr[:20]]
  ['Universal',
   'Declaration',
   'of',
   'Human',
   'Rights',
   'Preamble',
   'Whereas',
   'recognition']


Parts of Speech (POS)
*********************

Parts of Speech breaks down each word to their grammatical classification.

.. code:: python

  nltk.help.upenn_tagset('MD')

  text11 = "Children shouldn't drink a sugary drink before bed."
  text11 = nltk.word_tokenize(text11)
  print nltk.pos_tag(text13)

  [('Children', 'NNP'),
   ('should', 'MD'),
   ("n't", 'RB'),
   ('drink', 'VB'),
   ('a', 'DT'),
   ('sugary', 'JJ'),
   ('drink', 'NN'),
   ('before', 'IN'),
   ('bed', 'NN'),
   ('.', '.')]


Others
******
Other ways of feature extraction include using regular expression. The below example extracts different formats of dates.

.. code:: python

  import re
  import numpy as np
  import pandas as pd

  def function(x):
      # 04/20/2009; 04/20/09; 4/20/09; 4/3/09; 4-13-82
      if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', x) is not None:
          return re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', x).group()
      # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
      elif re.search(r'\d{,2}/\d{4}', x) is not None:
          return re.search(r'\d{,2}/\d{4}', x).group()
      # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
      elif re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z-.\s]*\d{,2}[-,\s]*\d{4}', x) is not None:
          return re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z-.\s]*\d{,2}[-,\s]*\d{4}', x).group().strip()
      # 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
      elif re.search(r'\d+\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.]*\s\d{4}', x) is not None:
          return re.search(r'\d+\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.]*\s\d{4}', x).group()
      # Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
      elif re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}\w{2},\s\d{4}', x) is not None:
          return re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}\w{2},\s\d{4}', x).group()
      # Feb 2009; Sep 2009; Oct 2010    
      elif re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}', x) is not None:
          return re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}', x).group()
      # 2009; 2010
      elif re.search(r'\d{4}', x) is not None:
          return re.search(r'\d{4}', x).group()
      else:
          return np.nan

  df2['new'] = df2['original'].apply(function)


Vectorization
--------------
Sklearn has several vectorizer functions that will tokenise and process text within the same function.
The process of vectorizing involves converting word characters into integers.
It has several important parameters, including
  * ``min_df``: e.g., 5. ignore items of minimum document frequency of 5 (can be integer or ratio)
  * ``max_df``: e.g., 0.2. ignore items of maximum document frequency of 5 (can be integer or ratio)
  * ``ngram_range``: e.g., (1-2). extracting only 1-grams or bigrams
  * ``stop_words``: list of stop words to remove
  * ``token_pattern``: e.g., '(?u)\\b\\w\\w\\w+\\b'). enter a regex pattern

CountVectorizer
****************
More from sklearn_.

.. _sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

.. code:: python

  # Using Count Vectorizer
  from sklearn.feature_extraction.text import CountVectorizer

  # Fit
  vect = CountVectorizer().fit(X_train)
  print vect.get_feature_names() # give a list of feature names

  X_train_vectorized = vect.transform(X_train)
  print vect.vocabulary_ # gives a dict of feature names with frequency
  print vect.vocabulary_.items() # gives pairs of key values in tuples instead, within a list


It is possible to fit & transform at the same time.

.. code:: python

  X_train_vectorized = CountVectorizer().fit_transform(X_train)


TfidVectorizer
**************
TF-IDF (Term Frequency-Inverse Document Frequency)
is a metric where high weight is given to terms that appear often in a particular document, 
but don't appear often in the corpus (all documents). 
Features with low tf–idf are either commonly used across all documents 
or rarely used and only occur in long documents.

TF-IDF can reduce the number of features required to train a model.

More from sklearn2_.

.. _sklearn2: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

.. code:: python

  from sklearn.feature_extraction.text import TfidfVectorizer
  # min_df, a minimum document frequency of < 5
  # extracting 1-grams and 2-grams
  vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)



