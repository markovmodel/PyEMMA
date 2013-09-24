'''
Created on 24.09.2013

@author: marscher
'''

# package import
try:
    import emma2
    emma2.msm.analyze.statdist(0)
    print "package import works"
except ImportError:
    print "could not import..."
    
# wild import
try:
    from emma2 import *
    msm.analyze.statdist(0)
    print "wild import works"
except ImportError:
    print "could not import..."
        
# direct import
try:
    from emma2.msm import analyze
    from emma2.msm.analyze.analyze import *
    analyze.statdist(0)
    statdist(0)
    print "direct import works"
except ImportError:
    print "could not import..."

if __name__ == '__main__':
    print "hi there"
    