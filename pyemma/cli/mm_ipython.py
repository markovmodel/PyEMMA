#!/usr/bin/env python
'''
mm_ipython
===========================
* install PyEMMA's IPython notebooks to a given location.
* run an IPython notebook server pointing to PyEmmas notebooks
'''
from __future__ import print_function

import sys
import argparse
import os
import errno


def handleArgs():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    default_dest = os.path.join(os.path.expanduser('~'), 'pyemma-ipython')

    parser.add_argument('--install', '-i', action="store_true")
    parser.add_argument('--dest', '-d', help='install location. Defaults to %s'
                        % default_dest, default=default_dest)

    parser.add_argument('--run', '-r', action="store_true",
                        help='run notebook server')

    parser.add_argument('--overwrite', '-f', action='store_true',
                        help='overwrite destination')
    parser.add_argument('--', nargs='+', dest='ipy_opt',
                        help="optional arguments to pass to 'ipython notebook'")

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = parser.parse_known_args()

    return args


def get_default_install_location():
    import pkg_resources
    # this gets the location of the most recent pyemma install
    dist = pkg_resources.get_distribution('pyemma')
    loc = os.path.join(dist.location, 'pyemma-ipython')
    return loc


def install_notebooks(dest, overwrite):
    import shutil
    src = get_default_install_location()
    print ("installing from %s to %s" % (src, dest))
    try:
        shutil.copytree(src, dest)
    except OSError as oe:
        if oe.errno == errno.EEXIST:
            if overwrite:
                print('overwriting given path %s' % dest)
                shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                raise RuntimeError('destination %s exists! If you are sure, '
                                   'you may want to overwrite it given "-f"'
                                   % dest)
    except Exception as e:
        raise RuntimeError('install destination %s could not be created. '
                           'Error was %s' % (dest, e))


def start_ipython_server(notebook_location, ipy_opt):
    """
    starts an ipython notebook server
    """
    import subprocess

    cmd = {'args': ['ipython', 'notebook', notebook_location]}

    if ipy_opt:
        cmd['args'] += ipy_opt

    if os.name == 'nt': # needed for windows to create background process
        cmd['creationflags'] = 0x00000008

    subprocess.Popen(**cmd)


def main():
    args, ipy_opt = handleArgs()

    # skip first arg of ipy_opt, if args are given
    if len(ipy_opt) >= 2:
        ipy_opt = ipy_opt[1:]

    if args.install:
        install_notebooks(args.dest, args.overwrite)
        if args.run:
            start_ipython_server(args.dest, ipy_opt)
    else:
        if args.run:
            start_ipython_server(get_default_install_location(), ipy_opt)

    return 0


if __name__ == '__main__':
    sys.exit(main())
