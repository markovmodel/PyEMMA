
Developer's Guide
=================

.. toctree::
   :maxdepth: 2

Dev Workflow
------------
Have a look at :doc:`DEVELOPMENT`.

Python imports
--------------
Relative imports (eg. you are in package A and import other modules from A)
should be avoided due to pylint

Commit messages
---------------

Prepend a tag named [$top_api_package] to commit message, if the files changed
mainly belongs to that api package.

Eg.: ::

      pyemma.msm.api.analysis => [msm/analysis]

This has two opportunitíes:

1. You know only be looking at a commit message text which package has been
changed and these changes propably has an influence on your current work.

2. Merged commits from feature branches nicely reintegrate in the whole bunch of
commits from other features in the devel or master branch (may be filtered very
fast).

Else you would have to look at the diff introduced by each commit (or filter
with gitk, which is somehow painful).

