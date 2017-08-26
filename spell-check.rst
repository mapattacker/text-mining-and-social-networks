Spell Checker
==============

.. code:: python

  from nltk.corpus import words
  correct_spellings = words.words()

Jaccard Distance on Trigram
----------------------------

.. code:: python

  def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
      # get first letter of each word with c
      c = [i for i in correct_spellings if i[0]=='c']
      # calculate the distance of each word with entry and link both together
      one = [(nltk.jaccard_distance(set(nltk.ngrams(entries[0], n=3)), \
                                    set(nltk.ngrams(a, n=3))), a) for a in c]

      i1 = [i for i in correct_spellings if i[0]=='i']
      two = [(nltk.jaccard_distance(set(nltk.ngrams(entries[1], n=3)), \
                                    set(nltk.ngrams(a, n=3))), a) for a in i1]

      v = [i for i in correct_spellings if i[0]=='v']
      three = [(nltk.jaccard_distance(set(nltk.ngrams(entries[2], n=3)), \
                                    set(nltk.ngrams(a, n=3))), a) for a in v]
      
      # sort them to ascending order so shortest distance is on top.
      # extract the word only
      output = [sorted(one)[0][1], sorted(two)[0][1], sorted(three)[0][1]]
      
      return output
      
  answer_nine()
  
Jaccard Distance on 4-gram
---------------------------

.. code:: python

  def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
      
      # get first letter of each word with c
      c = [i for i in correct_spellings if i[0]=='c']
      # calculate the distance of each word with entry and link both together
      one = [(nltk.jaccard_distance(set(nltk.ngrams(entries[0], n=4)), \
                                    set(nltk.ngrams(a, n=4))), a) for a in c]

      i1 = [i for i in correct_spellings if i[0]=='i']
      two = [(nltk.jaccard_distance(set(nltk.ngrams(entries[1], n=4)), \
                                    set(nltk.ngrams(a, n=4))), a) for a in i1]

      v = [i for i in correct_spellings if i[0]=='v']
      three = [(nltk.jaccard_distance(set(nltk.ngrams(entries[2], n=4)), \
                                    set(nltk.ngrams(a, n=4))), a) for a in v]
      
      # sort them to ascending order so shortest distance is on top.
      # extract the word only
      output = [sorted(one)[0][1], sorted(two)[0][1], sorted(three)[0][1]]
      
      return output
      
  answer_ten()
  
  
Edit Distance
--------------

.. code:: python

  def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):

      from nltk.corpus import words

      correct_spellings = words.words()

      # get first letter of each word with c
      c = [i for i in correct_spellings if i[0]=='c']
      # calculate the distance of each word with entry and link both together
      one = [((nltk.edit_distance(entries[0], a)), a) for a in c]

      i1 = [i for i in correct_spellings if i[0]=='i']
      two = [((nltk.edit_distance(entries[1], a)), a) for a in i1]

      v = [i for i in correct_spellings if i[0]=='v']
      three = [((nltk.edit_distance(entries[2], a)), a) for a in v]
      
      # sort them to ascending order so shortest distance is on top.
      # extract the word only
      output = [sorted(one)[0][1], sorted(two)[0][1], sorted(three)[0][1]]
      
      return output
      
  answer_ten()