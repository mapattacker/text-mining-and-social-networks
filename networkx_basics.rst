Graph Types
============

.. code:: python

  # Undirected Graph
  G = nx.Graph()
  G.add_edge('A','B')
  G.add_edge('B','C')
  
  
  # Directed Graph
  D = nx.Graph()
  D.add_edge('B','A')
  D.add_edge('B','C')
  
  
  # Multi-Graph
  M = nx.MultiGraph()
  M.add_edge('B','A')
  M.add_edge('B','C')

Edge Types
===========

.. code:: python

  # Weighted Edges
  W = nx.Graph()
  W.add_edge('A','B', weight=5)
  W.add_edge('B','C', weight=6)
  
  # Signed Edges 
  S = nx.Graph()
  S.add_edge('A','B', sign='+')
  S.add_edge('B','C', sign='-')

We can add edge attributes with any keys.

.. code:: python
  
  # Edge Attributes
  R = nx.Graph()
  R.add_edge('A','B', relation='friend')
  R.add_edge('B','C', relation='coworker')
  R.add_edge('B','D', relation='family')
  
  
Node Attributes
================

Same as edge attributes, nodes attributes can also be assigned with any keys.

.. code:: python

  G=nx.MultiGraph()
  G.add_node('A',role='manager')
  G.node['A']['role'] = 'team member'
  G.node['B']['role'] = 'engineer'


Joining Two Graphs
==================

Networkx can merge two graphs together with their differing weights when the edge list are the same.

.. code:: python

  new = nx.compose(a, b)
  
  name1	  name2	  weights
  Georgia	Lee	    {u'Weight': 10}
  Georgia	Claude	{u'weight': 3,u'Weight': 90}
  Georgia	Andy	  {u'weight': 1, u'Weight': -10}
  Georgia	Pablo	  {u'Weight': 0}
  Georgia	Frida	  {u'Weight': 0}
  Georgia	Vincent	{u'Weight': 0}
  Georgia	Joan	  {u'Weight': 0}
  Lee	    Claude	{u'Weight': 0}