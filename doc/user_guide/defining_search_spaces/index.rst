Defining search spaces
======================

In many cases, we may want to search a space of circuit architectures that is too large to be enumerated 
explicitly. In ``circuitree``, search spaces are specified implicitly as Grammars. The package comes with a few 
built-in Grammars for common use cases, and in other cases a custom Grammar can also be defined using the 
``CircuitGrammar`` low-level API.

.. toctree::
   :maxdepth: 1

   Built-in Grammars <built_in_grammars> 
   Custom Grammars <custom_grammars>