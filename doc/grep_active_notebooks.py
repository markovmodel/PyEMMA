# finds the notebooks referenced in ipython.rst and gets the paths to these notebooks
import re
import os


notebooks = []
with open('source/ipython.rst') as fh:
    for line in fh:
        match = re.match('\s+generated\/(.+)', line)
        if match:
            notebooks.append('{}.ipynb'.format(match.group(1)))


notebooks_paths = []
for root, dirs, files in os.walk('../pyemma-ipython'):
    for f in files:
        if f in notebooks:
            notebooks_paths.append(os.path.join(root, f))

assert len(notebooks) == len(notebooks_paths)
for n in notebooks_paths:
    print(n)
