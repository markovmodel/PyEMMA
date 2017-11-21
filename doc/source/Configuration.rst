
Runtime Configuration
=====================

You can change some runtime behaviour of PyEMMA by setting a configuration
value in PyEMMAs config module. These can be persisted to hard disk to be
permanent on every import of the package.

Examples
--------

Change values
^^^^^^^^^^^^^
To access the config at runtime eg. if progress bars should be shown:

>>> from pyemma import config # doctest: +SKIP
>>> print(config.show_progress_bars) # doctest: +SKIP
True
>>> config.show_progress_bars = False # doctest: +SKIP
>>> print(config.show_progress_bars) # doctest: +SKIP
False


Store your changes / Create a configuration directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create an editable configuration file, use the :py:func:`pyemma.config.save` method:

>>> from pyemma import config # doctest: +SKIP
>>> config.save('/tmp/pyemma_current.cfg') # doctest: +SKIP

This will store the current runtime configuration values in the given file.
Note that these settings will not be used on the next start of PyEMMA, because
you first need to tell us, where you have stored this file. To do so, please
set the environment variable **"PYEMMA_CFG_DIR"** to the directory, where you have
stored the config file.

* For Linux/OSX this thread `thread
  <https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables>`_
  may be helpful.
* For Windows have a look at
  `this <https://stackoverflow.com/questions/17312348/how-do-i-set-windows-environment-variables-permanently>`_.


For details have a look at the brief documentation:
https://docs.python.org/2/howto/logging.html

Default configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Default settings are stored in a provided pyemma.cfg file, which is included in
the Python package:

.. literalinclude:: ../../pyemma/pyemma.cfg
    :language: ini

Configuration files
^^^^^^^^^^^^^^^^^^^

To configure the runtime behavior such as the logging system or other parameters,
the configuration module reads several config files to build
its final set of settings. It searches for the file 'pyemma.cfg' in several
locations with different priorities:

#. $CWD/pyemma.cfg
#. $HOME/.pyemma/pyemma.cfg
#. ~/pyemma.cfg
#. $PYTHONPATH/pyemma/pyemma.cfg (always taken as default configuration file)

Note that you can also override the location of the configuration directory by
setting an environment variable named **"PYEMMA_CFG_DIR"** to a writeable path to
override the location of the config files.

The default values are stored in latter file to ensure these values are always
defined.

If no configuration file could be found, the defaults from the shipped package
will apply.


Load a configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to load a pre-saved configuration file, use the :py:func:`load` method:

>>> from pyemma import config # doctest: +SKIP
>>> config.load('pyemma_silent.cfg') # doctest: +SKIP


Configuration values
--------------------

.. autoclass:: pyemma.util._config.Config
    :members:
    :undoc-members:

    .. rubric:: Methods

    .. autoautosummary:: pyemma.util._config.Config
       :methods:

    .. rubric:: Attributes

    .. autoautosummary:: pyemma.util._config.Config
        :attributes:


Parallel setup
--------------

Some algorithms of PyEMMA use parallel computing. On one hand there is parallelisation due to
NumPy, which can use several threads to speed up raw NumPy computations. On the other hand PyEMMA itself
can start several threads and or sub-processes (eg. in clustering, MSM timescales computation etc.).

To limit the amount of threads/processes started by PyEMMA you can set the environment variable **PYEMMA_NJOBS**
to an integer value. This setting can also be overridden by the **n_jobs** property of the supported estimator.

To set the number of threads utilized by NumPy you can set the environment variable **OMP_NUM_THREADS**
to an integer value as well.

Note that this number will be multiplied by the setting for **PYEMMA_NJOBS**, if the the algorithm uses
multiple processes, as each process will use the same amount of OMP threads.

Setting these values too high, will lead to bad performance due to the overhead of maintaining multiple threads
and or processes.

By default `PYEMMA_NJOBS` will be chosen automatically to suit your hardware setup, but in shared environments
this can be sub-optimal.

For the popular SLURM cluster scheduler, we also respect the value of the environment variable **SLURM_CPUS_ON_NODE** and give it a high preference, if
`PYEMMA_NJOBS` is also set. So if you have chosen the number of CPUs for your cluster job, PyEMMA would then
automatically use the same amount of threads.
