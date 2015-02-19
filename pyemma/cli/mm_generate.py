#!/usr/bin/env python
# encoding: utf-8
"""
"""
import argparse
import sys
import os
from pyemma.util.log import getLogger
from pyemma.msm.generation import generate_traj
import pyemma

log = getLogger('mm_generate')


def handleArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=str, help='path to transition matrix.')
    parser.add_argument('-o', '--output', dest='output',
                        required=True, help='output filename of trajectory.')
    parser.add_argument('-dt', type=int, default=1)
    parser.add_argument('-steps', type=int)
    parser.add_argument('-start_state', type=int)

    args = parser.parse_args()

    return args


def main():
    args = handleArgs()

    try:
        _, ext = os.path.splitext(args.T)
        if ext == '.npy':
            pyemma.msm.io.load_matrix(args.T)
        else:
            T = pyemma.msm.io.read_matrix(args.T)
    except IOError:
        log.error('error during reading transition matrix file %s' % args.T)

    traj = generate_traj(T, args.start_state, args.steps, args.dt)

    try:
        pyemma.msm.io.save_matrix(args.output, traj)
    except IOError:
        log.exception("error during saving resulting trajectory to %s"
                      % args.output)

if __name__ == '__main__':
    sys.exit(main())
