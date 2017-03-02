import os
import sys
import pytest
import shutil
import tempfile

test_pkg = 'pyemma'
cover_pkg = test_pkg

junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', '.'), 'junit.xml')
print("cwd:", os.getcwd())
pytest_cfg = 'setup.cfg'
conftest = 'conftest.py'

print("Using pytest config file: %s" % pytest_cfg)
assert os.path.exists(pytest_cfg), "pytest cfg not found"
tmp = tempfile.mkdtemp()
shutil.copy(pytest_cfg, tmp)
shutil.copy(conftest, tmp)
os.chdir(tmp)
print("current cwd:", os.getcwd())

# matplotlib headless backend
with open('matplotlibrc', 'w') as fh:
    fh.write('backend: Agg')

pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=html "
               "--doctest-modules "
               #"-n 2 "# -p no:xdist" # disable xdist in favour of coverage plugin
               "--junit-xml={junit_xml} "
               "-c {pytest_cfg} --fixtures"
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml, pytest_cfg=pytest_cfg)
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

# copy it to home, so we can process it with codecov etc.
shutil.copy('coverage.xml', os.path.expanduser('~/'))

sys.exit(res)

