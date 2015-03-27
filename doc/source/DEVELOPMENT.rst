=================
Developer's Guide
=================

.. toctree::
   :maxdepth: 2

Basic Idea:
-----------
Use a development branch to work on and reintegrate this periodically
into the master branch. To develop certain features one should use
feature branches, which are themselves branched from the development
branch. Feature branches will be reintegrated into the development
branch.  For releases we merge the development branch into the master
branch.

Why:
----
* Have a tested, "stable" master branch, which is directly available by cloning.
* Interference with other devs is minimized until merging.


How:
----
A maintainer merges the development branch(es) periodically. These
branches should be pretested (e.g by automatic runs of the test-suite
and tagging these branches as "tested"/"stable" etc.). This can be
achieved by a continous integration (CI) software like Jenkins
http://jenkins-ci.org or Travis-CI http://travis-ci.org, the first one
is open source and the second one is free for open source projects
only.


Commit messages
---------------

Prepend a tag named [$top_api_package] to commit message, if the files
changed mainly belongs to that api package.

Eg.: ::

      pyemma.msm.api.analysis => [msm/analysis]

This has two opportunitíes:

1. You know only be looking at a commit message text which package has
been changed and these changes propably has an influence on your
current work.

2. Merged commits from feature branches nicely reintegrate in the
whole bunch of commits from other features in the devel or master
branch (may be filtered very fast).

Else you would have to look at the diff introduced by each commit (or
filter with gitk, which is somehow painful).


Testing
-------
We use Pythons unittest module to write implement test cases for every algorithm.

To run all tests invoke: ::

    python setup.py test

or directly invoke nosetests in pyemma working copy: ::

    nosetests $PYEMMA_DIR

despite running all tests (which is encouraged if you are changing core features),
you can run individual tests by directly invoking them with python interpreter.


Workflow for branching and merging:
-----------------------------------
A developer creates a feature branch "feature" and commits his work to
this branch. When he is done with his work (have written at least a
working test case for it), he pushes this feature branch to his fork
and creates a pull request.  The pull request can then be reviewed and
merged upstream.

0. Get up to date - pull the latest changes from devel

::
   
      # first get the lastest changes
      git pull 

1. Compile extension modules (works also for conda)

::

      python setup.py develop

The develop install has the advantage that if only python scripts are
being changed eg. via an pull or a local edit, you do not have to
reinstall anything, because the setup command simply created a link to
your working copy.

2. branch 'feature' from the devel branch and switch to it:

::
   
      # switch to development branch
      git checkout devel 
      # create new branch and switch to it
      git checkout -b feature 

3. work on the newly created branch:

::

      touch fancy_feat.py

4. Write unit test and TEST IT! :-)

::

      touch fancy_feat_test.py
      # test the unit
      python fancy_feat_test.py
      # run the whole test-suite 
      # (to ensure that your newfeature has no side-effects)
      cd $PYEMMA_DIR
      python setup.py test
      

5. commit your changes 

::

      git commit fancy_feat.py fancy_feat_test.py \
          -m "Implementation and unit test for fancy feature"


6. Make changes available by creating a pull request

::

      # push your branch to your fork on github
      git push myfork feature
      
On github create a pull request from myfork/feature to origin/devel,
see https://help.github.com/articles/using-pull-requests


Conclusions:
------------

* Working branches do not interfere with other ones.
* The devel branch contains all tested implemented features
* The devel branch is used to test for interference of features
* Work with pull request to ensure your features are being tested by CI 
* The master branch contains all tested features and represents the
  set of features that are suitable for public usage
