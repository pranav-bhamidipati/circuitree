CircuiTree: Biochemical circuit design using RL
===============================================

``circuitree`` is a Python package for optimizing the architecture, or topology, of a biochemical network 
for a particular behavior using reinforcement learning (RL), specifically Monte carlo Tree Search 
(MCTS). Once the user defines a space of topologies they want to search and supplies a reward function, 
``circuitree`` will search for the best topologies. 

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/tutorial_1_getting_started

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/core_concepts
   user_guide/conventions
   user_guide/tutorial_2_mcts_in_parallel
   user_guide/defining_search_spaces/index
   user_guide/api

.. toctree::
   :maxdepth: 1
   :caption: Credits and citation

   credits_and_citation.rst
