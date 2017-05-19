
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
