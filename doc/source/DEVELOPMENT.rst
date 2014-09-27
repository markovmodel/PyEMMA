Development workflow
====================

Basic Idea:
-----------
Use a development branch to work on and reintegrate this periodically into the
master branch. To develop certain features one should use feature branches,
which are themselves branched from the development branch. Feature branches
will be reintegrated into the development branch.
For releases we merge the development branch into the master branch.

Why:
----
* Have a tested, "stable" master branch, which is directly available by cloning.
* Interference with other devs is minimized until merging.


How:
----
A maintainer merges the development branch(es) periodically. These branches
should be pretested (e.g by automatic runs of the test-suite and tagging these
branches as "tested"/"stable" etc.). This can be achieved by a continous
integration (CI) software like Jenkins (http://jenkins-ci.org) or Travis-CI
(http://travis-ci.org), the first one is open source and the second one is free
for open source projects only.

Workflow for branching and merging:
-----------------------------------
A dev creates a feature branch "impl_fancy_feature" and pushes his work to this
branch. When he is done with his fancy feature (have written at least a working
test case for it), he pushes this work to his feature branch and merges it to
the development branch.

1. create a branch from the development branch and switch to it:

::

      git pull # first get the lastest changes
      git checkout development # switch to development branch
      git branch impl_fancy_feature # create new branch
      git checkout impl_fancy_feature # switch to new branch

2. work on the newly created branch:

::

      touch fancy_feat.py
      git commit fancy_feat.py -m"Just created source for fancy_feature"

3. TEST IT! :-)

::

      touch fancy_feat_test.py
      git commit fancy_feat_test.py -m"This is the unit test for fancy_feat"

4. reintegrate your work with the development branch

::

      # push your branch to the origin (remote repo) to backup it
      git push origin impl_fancy_feature
      # switch to devel branch 
      git checkout development
      # merge your fancy feature with devel branch
      git merge impl_fancy_feature


Conclusions:
------------

* Working branches do not interfere with other ones.
* The development branch contains all tested implemented features
* The development branch is used to test for interference of features
* The master branch contains all tested features and represents the
  set of features that are suitable for public usage
