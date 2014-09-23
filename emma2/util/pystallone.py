"""
This module wraps the initialization process of the pystallone package. As soon
as the module gets imported a global Java Virtual Machine (jvm) instance is 
being created. The parameters for the jvm are red from a emma2.cfg file (see 
emma2.util.config for more details).

 
The API variable is the main entry point into the Stallone API factory.

Examples
--------

create a double vector and assigning values:
>>> from emma2.util.pystallone import stallone as st
>>> x = st.api.API.doublesNew.array(10) # create double array with 10 elements
>>> x.set(5, 23.0) # set index 5 to 23.0
>>> print(x)
0.0     0.0     0.0     0.0     0.0     23.0     0.0     0.0     0.0     0.0

equivalently:
>>> x = emma2.util.pystallone.API.doublesNew.array(10)
...

Created on 15.10.2013
package externalized on 8.8.2014

@author: marscher
"""
from __future__ import absolute_import
from pystallone import startJVM

def _get_jvm_args():
    """
    reads in the configuration values for the java virtual machine and 
    returns a list containing them all.
    """
    from .config import configParser
    initHeap = '-Xms%s' % configParser.get('Java', 'initHeap')
    maxHeap = '-Xmx%s' % configParser.get('Java', 'maxHeap')
    optionalArgs = configParser.get('Java', 'optionalArgs')
    optional_cp = configParser.get('Java', 'classpath')
    return [initHeap, maxHeap, optionalArgs, optional_cp]

try:
    startJVM(None, _get_jvm_args())
except:
    from .log import getLogger
    _log = getLogger()
    _log.exception("jvm startup failed")
    raise

# after we have successfully started the jvm, we import the rest of the symbols.
from pystallone import *