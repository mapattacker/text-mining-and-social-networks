Text Classificaton
==================


Add New Features to Vectorizer
------------------------------

.. code:: python

  def add_feature(X, feature_to_add):
      """
      Returns sparse feature matrix with added feature.
      feature_to_add can also be a list of features.
      """
      from scipy.sparse import csr_matrix, hstack
      return hstack([X, csr_matrix(feature_to_add).T], 'csr')
  
  
  # add character count feature to dataset
  x_len = X_train.apply(len)
  X_train_aug = add_feature(X_train_vectorized, x_len)

  x_len2 = X_test.apply(len)
  X_test_aug = add_feature(X_test_vectorized, x_len2)
    

Multi-Nominal Naive Bayes & CountVectorizer
--------------------------------------------

Using plain Naive Bayes & CountVectorizer only.

.. code:: python

  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.metrics import roc_auc_score
  

  spam_data = pd.read_csv('spam.csv')
  spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
  spam_data.head(10)

  X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                      spam_data['target'], random_state=0)
                                                      
  # vectorise text
  vect = CountVectorizer().fit(X_train)
  X_train_vectorized = vect.transform(X_train)

  # train model
  model = MultinomialNB(alpha=0.1)
  model.fit(X_train_vectorized, y_train)

  # Predict the transformed test documents
  predictions = model.predict(vect.transform(X_test))

  roc = roc_auc_score(y_test, predictions)
  
  return roc
  # 0.97208121827411165
  

Multi-Nominal Naive Bayes & TfidfVectorizer
--------------------------------------------

Now using Naive Bayes & TfidfVectorizer only.

.. code:: python
  
  def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    
    # Predict the transformed test documents
    predictions = model.predict(vect.transform(X_test))
    roc = roc_auc_score(y_test, predictions)
    
    return roc
    # 0.94162436548223349
    

Support Vector Machine & TfidfVectorizer
--------------------------------------------

Using SVM & TfidfVectorizer, and also add an a new feature, character count.

.. code:: python

  from sklearn.svm import SVC
  from sklearn.metrics import roc_auc_score

  def answer_seven():
      
      # vectorise & transform
      vect = TfidfVectorizer(min_df=5).fit(X_train)
      X_train_vectorized = vect.transform(X_train)
      X_test_vectorized = vect.transform(X_test)

      # add character count feature to dataset
      x_len = X_train.apply(len)
      X_train_aug = add_feature(X_train_vectorized, x_len)

      x_len2 = X_test.apply(len)
      X_test_aug = add_feature(X_test_vectorized, x_len2)

      # fit model
      model = SVC(C=10000).fit(X_train_aug, y_train)

      # predictions
      predictions = model.predict(X_test_aug)

      roc = roc_auc_score(y_test, predictions)
      
      return roc
  # 0.95813668234215565
      

Logistic Regression & TfidfVectorizer
--------------------------------------
Add two more features, character count & digit count.


.. code:: python

    from sklearn.linear_model import LogisticRegression

    def answer_nine():
        
        # vectorise & transform
        vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
        X_train_vectorized = vect.transform(X_train)
        X_test_vectorized = vect.transform(X_test)

        # add character count feature to dataset
        x_len = X_train.apply(len)
        X_train_aug = add_feature(X_train_vectorized, x_len)
        # add number of digits
        x_digit = X_train.apply(lambda x: len(re.sub('\D','', x)))
        X_train_aug2 = add_feature(X_train_aug, x_digit)

        x_len2 = X_test.apply(len)
        X_test_aug = add_feature(X_test_vectorized, x_len2)
        x_digit2 = X_test.apply(lambda x: len(re.sub('\D','', x)))
        X_test_aug2 = add_feature(X_test_aug, x_digit2)

        # fit model
        model = LogisticRegression(C=100).fit(X_train_aug2, y_train)

        # predictions
        predictions = model.predict(X_test_aug2)

        roc = roc_auc_score(y_test, predictions)  
        
        return roc
        
    # 0.97040897747143606
        
Logistic Regression & TfidfVectorizer
--------------------------------------
Add three more features, character count, digit count & non-word count.


.. code:: python

  import re

  def answer_eleven():

      # vectorise & transform
      vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
      X_train_vectorized = vect.transform(X_train)
      X_test_vectorized = vect.transform(X_test)

      # add character count feature to train dataset
      x_len = X_train.apply(len)
      X_train_aug = add_feature(X_train_vectorized, x_len)
      # add number of digits
      x_digit = X_train.apply(lambda x: len(re.sub('\D','', x)))
      X_train_aug2 = add_feature(X_train_aug, x_digit)
      # add number of non-word
      x_nword = X_train.apply(lambda x: len(re.sub('\w','', x)))
      X_train_aug3 = add_feature(X_train_aug2, x_nword)



      # repeat for test dataset
      x_len2 = X_test.apply(len)
      X_test_aug = add_feature(X_test_vectorized, x_len2)
      x_digit2 = X_test.apply(lambda x: len(re.sub('\D','', x)))
      X_test_aug2 = add_feature(X_test_aug, x_digit2)
      x_nword2 = X_test.apply(lambda x: len(re.sub('\w','', x)))
      X_test_aug3 = add_feature(X_test_aug2, x_nword2)


      # fit model
      model = LogisticRegression(C=100).fit(X_train_aug3, y_train)

      # predictions & AUC
      predictions = model.predict(X_test_aug3)
      roc = roc_auc_score(y_test, predictions) 

      # get the feature names as numpy array
      feature_names = np.array(vect.get_feature_names()).tolist()
      # add the 3 new features into the array
      feature_names.extend(['length_of_doc', 'digit_count', 'non_word_char_count'])
      feature_names = np.array(feature_names)

      # sorted model coeff
      sorted_coef_index = model.coef_[0].argsort()

      small = feature_names[sorted_coef_index[:10]].tolist()
      large = feature_names[sorted_coef_index[:-11:-1]].tolist()

      output=(roc, small, large)

      return output
    
  (0.97885931107074342,
   [u'. ', u'..', u' i', u'? ', u' y', u' go', u':)', u' h', u'he', u'h '],
   [u'digit_count',
    u'ne',
    u'co',
    u'ia',
    u'xt',
    u'mob',
    u'ww',
    u' x',
    u' ch',
    u'ar'])