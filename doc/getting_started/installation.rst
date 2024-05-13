.. _installation:

Installation
============

For parallel and/or distributed computing

	$ pip install --upgrade circuitree[distributed]

For basic usage

	$ pip install --upgrade circuitree

Dependencies
------------

`circuitree` has the following dependencies.

- Python 3.10 or newer
- Numpy_
- NetworkX_
- Pandas_
- SciPy_

For progress bars, `circuitree` uses the tqdm_ package, which is optional.

`circuitree[distributed]` additionally uses the following dependencies to manage a distributed computing environment.

- Celery_
- gevent_

.. _NumPy: http://www.numpy.org/
.. _NetworkX: https://networkx.org/
.. _Pandas: http://pandas.pydata.org
.. _SciPy: https://www.scipy.org/
.. _tqdm: https://github.com/tqdm/tqdm
.. _Celery: https://docs.celeryq.dev/en/stable/
.. _gevent: https://www.gevent.org/