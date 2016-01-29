# Copyright (c) 2016 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as _np

__all__ = [
    'get_umbrella_sampling_parameters',
    'get_umbrella_bias_sequences',
    'get_averaged_bias_matrix']

def _ensure_umbrella_center(candidate, dimension):
    if isinstance(candidate, (_np.ndarray)):
        assert candidate.ndim == 1
        assert candidate.shape[0] == dimension
        return candidate.astype(_np.float64)
    elif isinstance(candidate, (int, long, float)):
        return candidate * _np.ones(shape=(dimension,), dtype=_np.float64)
    else:
        raise TypeError("unsupported type")

def _ensure_force_constant(candidate, dimension):
    if isinstance(candidate, (_np.ndarray)):
        assert candidate.shape[0] == dimension
        if candidate.ndim == 2:
            assert candidate.shape[1] == dimension
            return candidate.astype(_np.float64)
        elif candidate.ndim == 1:
            return _np.diag(candidate).astype(dtype=_np.float64)
        else:
            raise TypeError("usupported shape")
    elif isinstance(candidate, (int, long, float)):
        return candidate * _np.ones(shape=(dimension, dimension), dtype=_np.float64)
    else:
        raise TypeError("unsupported type")

def get_umbrella_sampling_parameters(
    us_trajs, us_centers, us_force_constants, md_trajs=None, kT=None):
    umbrella_centers = []
    force_constants = []
    ttrajs = []
    nthermo = 0
    for i in range(len(us_trajs)):
        state = None
        this_center = _ensure_umbrella_center(
            us_centers[i], us_trajs[i].shape[1])
        this_force_constant = _ensure_force_constant(
            us_force_constants[i], us_trajs[i].shape[1])
        for j in range(nthermo):
            if _np.all(umbrella_centers[j] == this_center) and \
                _np.all(force_constants[j] == this_force_constant):
                state = j
                break
        if state is None:
            umbrella_centers.append(this_center.copy())
            force_constants.append(this_force_constant.copy())
            ttrajs.append(nthermo * _np.ones(shape=(us_trajs[i].shape[0],), dtype=_np.intc))
            nthermo += 1
        else:
            ttrajs.append(state * _np.ones(shape=(us_trajs[i].shape[0],), dtype=_np.intc))
    if md_trajs is not None:
        umbrella_centers.append(
            _np.zeros(shape=umbrella_centers[-1].shape, dtype=umbrella_centers[-1].dtype))
        force_constants.append(
            _np.zeros(shape=force_constants[-1].shape, dtype=force_constants[-1].dtype))
        for md_traj in md_trajs:
            ttrajs.append(nthermo * _np.ones(shape=(md_traj.shape[0],), dtype=_np.intc))
        nthermo += 1
    umbrella_centers = _np.array(umbrella_centers, dtype=_np.float64)
    force_constants = _np.array(force_constants, dtype=_np.float64)
    if kT is not None:
        assert isinstance(kT, (int, long, float))
        assert kT > 0.0
        force_constants /= kT
    return ttrajs, umbrella_centers, force_constants

def get_umbrella_bias_sequences(trajs, umbrella_centers, force_constants):
    from thermotools.util import get_umbrella_bias as _get_umbrella_bias
    bias_sequences = []
    for traj in trajs:
        bias_sequences.append(
            _get_umbrella_bias(traj, umbrella_centers, force_constants))
    return bias_sequences

def get_averaged_bias_matrix(bias_sequences, dtrajs, nstates=None):
    from thermotools.util import logsumexp as _logsumexp
    from thermotools.util import logsumexp_pair as _logsumexp_pair
    from thermotools.util import kahan_summation as _kahan_summation
    nmax = int(_np.max([dtraj.max() for dtraj in dtrajs]))
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    nthermo = bias_sequences[0].shape[1]
    bias_matrix = -_np.ones(shape=(nthermo, nstates), dtype=_np.float64) * _np.inf
    counts = _np.zeros(shape=(nstates,), dtype=_np.intc)
    for s in range(len(bias_sequences)):
        for i in range(nstates):
            idx = (dtrajs[s] == i)
            nidx = idx.sum()
            if nidx == 0:
                continue
            counts[i] += nidx
            selected_bias_sequence = bias_sequences[s][idx, :]
            for k in range(nthermo):
                bias_matrix[k, i] = _logsumexp_pair(
                    bias_matrix[k, i],
                    _logsumexp(
                        _np.ascontiguousarray(-selected_bias_sequence[:, k]),
                        inplace=False))
    return _np.log(counts)[_np.newaxis, :] - bias_matrix
