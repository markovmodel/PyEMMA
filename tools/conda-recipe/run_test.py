import subprocess
import os
import sys

src_dir = os.getenv('SRC_DIR')

nose_run = "nosetests pyemma -vv --with-coverage --cover-inclusive --cover-package=pyemma" \
           " --with-doctest --doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS".split(' ')
res = subprocess.call(nose_run, cwd=src_dir)
sys.exit(res)
