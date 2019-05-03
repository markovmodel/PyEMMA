#!/usr/bin/env bash

set +x

virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
which pip
which pip3

pip install .

