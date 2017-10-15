Loading Data as Graph
=====================

.. code:: python

  import networkx as nx

Format
------

Adjacency List
***************
If the data is in an adjacency list, it will appear like below. 
The left most represents nodes, and others on its right represents nodes that are linked to it.

.. code:: python

  0 1 2 3 5
  1 3 6
  2
  3 4
  4 5 7
  5 8
  6
  7
  8 9
  9
  
To call it from a file, we use ``nx.read_adlist``.

.. code:: python

  G2 = nx.read_adjlist('G_adjlist.txt', nodetype=int)
  G2.edges()
  
  [(0, 1),
   (0, 2),
   (0, 3),
   (0, 5),
   (1, 3),
   (1, 6),
   (3, 4),
   (5, 4),
   (5, 8),
   (4, 7),
   (8, 9)]

Edge List
***************
Edge list is just a two column representation of one node to another. It can have additional columns for weights.

.. code:: python

  [(0, 1, {'weight': 4}),
   (0, 2, {'weight': 3}),
   (0, 3, {'weight': 2}),
   (0, 5, {'weight': 6}),
   (1, 3, {'weight': 2}),
   (1, 6, {'weight': 5}),
   (3, 4, {'weight': 3}),
   (5, 4, {'weight': 1}),
   (5, 8, {'weight': 6}),
   (4, 7, {'weight': 2}),
   (8, 9, {'weight': 1})]

We can use ``nx.read_edgelist()`` to transform it into a graph network.

.. code:: python

  G4 = nx.read_edgelist('G_edgelist.txt', data=[('Weight', int)], , delimiter='\t')
  

Adjacency Matrix
*****************  

From a graph network, we can transform it into an adjacency matrix using a pandas dataframe.

.. code:: python

  import pandas as pd

  nx.to_pandas_dataframe(g, weight='distance')
  
        1.0	    2.0	    3.0	    4.0	    5.0	  6.0	    7.0     
  1.0	  0.0   	1306.0	0.0	    0.0	  2161.0	2661.0	0.0
  2.0	  1306.0	0.0	    919.0	  629.0	0.0	    0.0	    0.0
  3.0	  0.0	    919.0	  0.0	    435.0	1225.0	0.0	    1983.0
  4.0	  0.0	    629.0	  435.0	  0.0	  0.0	    0.0	    0.0
  5.0	  2161.0	0.0	    1225.0	0.0	  0.0	    1483.0	1258.0
  6.0	  2661.0	0.0	    0.0	    0.0	  1483.0	0.0	    0.0
  7.0	  0.0	    0.0	    1983.0	0.0  	1258.0	0.0   	0.0

An adjacency matrix can also be loaded back to a graph

.. code:: python

  G3 = nx.Graph(matrix)
  G3.edges()


SQL > DataFrame > Graph
------------------------

The below code uses an edge list format.

.. code:: python

  import psycopg2
  import pandas as pd
  
  conn = psycopg2.connect(database="postgres", user="postgres", password="***", host="127.0.0.1", port="5432")

  query = """SELECT fromnode, tonode, distance from edges"""
  df = pd.read_sql_query(query, conn)
  g = nx.from_pandas_dataframe(df, 'fromnode', 'tonode', 'distance') # or edge_attr='distance'


Graph > DataFrame
------------------

Sometimes, it is necessary to convert a graph into an edge list into a dataframe to utilise pandas 
powerful analysis abilities.

.. code:: python

  df = pd.DataFrame(new.edges(data=True), columns=['name1','name2','weights'])
  df['relation'] = df['weights'].map(lambda x: x['Weight'])
  
  name1	  name2	  weights
  Georgia	Lee	    {u'Weight': 10}
  Georgia	Claude	{u'weight': 3, u'Weight': 90}
  Georgia	Andy	  {u'weight': 1, u'Weight': -10}
  Georgia	Pablo	  {u'Weight': 0}
  Georgia	Frida	  {u'Weight': 0}
  Georgia	Vincent	{u'Weight': 0}
  Georgia	Joan	  {u'Weight': 0}
  Lee	    Claude	{u'Weight': 0}
  Lee	    Andy	


Printing out data
------------------

.. code:: python

  # list nodes
  g.nodes()
  # list edges
  g.edges()
  # show all data, including weights and attributes
  g.nodes(data=True)
  g.edges(data=True)
  # number of edges / nodes
  len(g) # or g.number_of_nodes()
  g.number_of_edges()

