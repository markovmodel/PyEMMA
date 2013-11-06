'''
Intended to define global arguments in EMMA2.
If you want to define a subparser, add it to parser instance defined here.

Code:
    import argparse
    import emma2.util.args as args
    subparser = argparse.ArgumentParser(parents=[args.root_parser], ...)


Created on 05.11.2013

@author: marscher
'''

import argparse

__all__ = ['parser']

root_parser = argparse.ArgumentParser(prog="Emma2")
