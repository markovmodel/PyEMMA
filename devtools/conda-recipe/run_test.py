import os
import sys
import tempfile

import pytest

test_pkg = 'pyemma'
cover_pkg = test_pkg

# where to write junit xml
xml_results_dest = os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir())
assert os.path.isdir(xml_results_dest), 'no dest dir available'
target_dir = os.path.dirname(xml_results_dest)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

junit_xml = os.path.join(xml_results_dest, 'junit.xml')
cov_xml = os.path.join(xml_results_dest, 'coverage.xml')

print('junit destination:', junit_xml)
# njobs_args = '-p no:xdist' if os.getenv('TRAVIS') or os.getenv('CIRCLECI') else '-n2'

pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               "--junit-xml={junit_xml} "
               "-c {pytest_cfg}"
               "--durations=20 "
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml, pytest_cfg='setup.cfg',
                       dest_report=cov_xml)
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

sys.exit(res)

