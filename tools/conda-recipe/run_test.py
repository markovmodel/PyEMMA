
import subprocess
import os
import sys
import shutil
import re

src_dir = os.getenv('SRC_DIR')


# matplotlib headless backend
with open('matplotlibrc', 'w') as fh:
    fh.write('backend: Agg')


def coverage_report():
    fn = '.coverage'
    assert os.path.exists(fn)
    build_dir = os.getenv('TRAVIS_BUILD_DIR')
    dest = os.path.join(build_dir, fn)
    print( "copying coverage report to", dest)
    shutil.copy(fn, dest)
    assert os.path.exists(dest)

    # fix paths in .coverage file
    with open(dest, 'r') as fh:
        data = fh.read()
    match= '"/home/travis/miniconda/envs/_test/lib/python.+?/site-packages/.+?/(pyemma/.+?)"'
    repl = '"%s/\\1"' % build_dir
    data = re.sub(match, repl, data)
    os.unlink(dest)
    with open(dest, 'w+') as fh:
       fh.write(data)

nose_run = "nosetests pyemma -vv" \
           " --with-coverage --cover-inclusive --cover-package=pyemma" \
           " --with-doctest --doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS" \
           .split(' ')

res = subprocess.call(nose_run)


# move .coverage file to git clone on Travis CI
if os.getenv('TRAVIS', False):
   coverage_report()

sys.exit(res)

