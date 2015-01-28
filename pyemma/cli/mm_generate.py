#!/usr/bin/env python
# encoding: utf-8
"""
"""
import argparse
import sys

from pyemma.msm.generation import trajectory_generator

def handleArgs():
# -o <filename>\n" 
# "[-sigma <double>:{0.6}]\n"
# + "[-dt <double>:{0.1}]\n" 
# "[-randomseed <int>]\n" 
#" -steps <int>\n" +
# " -potdef <filename>\n" 
# " -start <int> <int>\n";
            
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--output', dest='output', required=True, help='output filename of trajectory.')
    parser.add_argument('-sigma', default=0.6)
    parser.add_argument('-dt', default=0.1)
    parser.add_argument('-randomseed', type=int)
    parser.add_argument('-steps', type=int)
    parser.add_argument('-potdef')
    parser.add_argument('-start')
    
    args = parser.parse_args()
    
    # perfom sanity checks
    
    return args


def main():
    args = handleArgs()
    
    trajectory_generator(T, start, stop)
    
    raise NotImplementedError

if __name__ == '__main__':
    sys.exit(main())