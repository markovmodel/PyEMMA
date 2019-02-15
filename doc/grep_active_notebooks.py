# finds the notebooks referenced in ipython.rst and gets the paths to these notebooks
import re
import os
from glob import glob

_notebooks = []
with open('source/ipython.rst') as fh:
    for line in fh:
        match = re.match('\s+generated\/(.+)', line)
        if match:
            _notebooks.append('{}.ipynb'.format(match.group(1)))


notebooks_paths = []
# legacy tutorials
for root, dirs, files in os.walk('./pyemma-ipython'):
    for f in files:
        if f in _notebooks:
            notebooks_paths.append(os.path.join(root, f))

# live coms tutorials.
tutorial_nbs = sorted(glob('tutorials/notebooks/*.ipynb'))
assert tutorial_nbs

notebooks_paths += tutorial_nbs

if __name__ == '__main__':
    #assert len(_notebooks) == len(notebooks_paths)
    for n in notebooks_paths:
        print(n)
