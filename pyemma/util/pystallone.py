"""
This module wraps the initialization process of the pystallone package. As soon
as the module gets imported a global Java Virtual Machine (jvm) instance is
being created. The parameters for the jvm are red from a pyemma.cfg file (see
pyemma.util.config for more details).


The API variable is the main entry point into the Stallone API factory.

Examples
--------

create a double vector and assigning values:
>>> from pyemma.util.pystallone import stallone as st
>>> x = st.api.API.doublesNew.array(10) # create double array with 10 elements
>>> x.set(5, 23.0) # set index 5 to 23.0
>>> print(x)
0.0     0.0     0.0     0.0     0.0     23.0     0.0     0.0     0.0     0.0

equivalently:
>>> x = pyemma.util.pystallone.API.doublesNew.array(10)
...

Created on 15.10.2013
package externalized on 8.8.2014

@author: marscher
"""
from __future__ import absolute_import
from pystallone import startJVM
from .log import getLogger as _getLogger
_log = _getLogger()


def _get_jvm_args():
    """
    reads in the configuration values for the java virtual machine and 
    returns a list containing them all.
    """
    from pyemma.util.config import conf_values
    java_conf = conf_values['Java']
    initHeap = '-Xms%s' % java_conf.initheap
    maxHeap = '-Xmx%s' % java_conf.maxheap
    # optargs may contain lots of options separated by whitespaces, need a list
    optionalArgs = java_conf.optionalargs.split()
    optional_cp = "-Djava.class.path=%s" % java_conf.classpath
    return [initHeap, maxHeap, optional_cp] + optionalArgs

try:
    args = _get_jvm_args()
    startJVM(None, args)
except:
    _log.exception("jvm startup failed")
    raise

# after we have successfully started the jvm, we import the rest of the symbols.
from pystallone import *