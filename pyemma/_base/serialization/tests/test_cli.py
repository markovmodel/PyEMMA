
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import shutil
import tempfile

import pytest
from numpy.testing import assert_

from pyemma._base.serialization.cli import main
from pyemma.coordinates import source, tica, cluster_kmeans


@pytest.fixture
def model_file():
    file = None
    try:
        from pyemma.datasets import get_bpti_test_data
        d = get_bpti_test_data()
        trajs, top = d['trajs'], d['top']
        s = source(trajs, top=top)

        t = tica(s, lag=1)

        c = cluster_kmeans(t)
        file = tempfile.mktemp()
        c.save(file, save_streaming_chain=True)

        yield file
    finally:
        if file is not None:
            shutil.rmtree(file, ignore_errors=True)


def test_recursive(model_file):
    """ check the whole chain has been printed"""
    from pyemma.util.contexts import Capturing
    with Capturing() as out:
        main(['--recursive', model_file])
    assert out
    all_out = '\n'.join(out)
    assert_('FeatureReader' in all_out)
    assert_('TICA' in all_out)
    assert_('Kmeans' in all_out)
