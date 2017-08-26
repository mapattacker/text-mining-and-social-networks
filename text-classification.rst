Text Classificaton
==================

Multi-Nominal Naive Bayes & CountVectorizer
--------------------------------------------

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
  

Support Vector Machine & TfidfVectorizer
--------------------------------------------

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