import subprocess
import os
import sys
import shutil

src_dir = os.getenv('SRC_DIR')

# matplotlib headless backend
with open('matplotlibrc', 'w') as fh:
    fh.write('backend: Agg')

nose_run = "nosetests pyemma -vv --with-coverage --cover-inclusive --cover-package=pyemma" \
           " --with-doctest --doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS".split(' ')
res = subprocess.call(nose_run)

# move .coverage file to git clone on Travis CI
if os.getenv('TRAVIS', False):
    fn = '.coverage'
    assert os.path.exists(fn)
    dest = os.path.join(os.getenv('TRAVIS_BUILD_DIR'), fn)
    print( "copying coverage report to", dest)
    shutil.copy(fn, dest)
    assert os.path.exists(dest)
sys.exit(res)
