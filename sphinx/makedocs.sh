export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

PYTHONPATH=$( cd .. && pwd ) sphinx-autogen -o generated api.rst

make html
make html
