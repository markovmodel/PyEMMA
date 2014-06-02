Testing
=======

You have two possibilities to test emma2:

To run the tests provided in Emma2 you can either use tox, which provides an
isolated testing environment (with all needed dependencies) or you can directly
run single tests from their locations on the command line.

In the emma2 root directory of your repository execute:

::

   python test/testsuite.py


or use tox, which will take some time, since it first creates an isolated env.

::

   tox -e unit_test


Both commands return 0 on success, so they can be used for git bisect and other
shell scripting purposes.