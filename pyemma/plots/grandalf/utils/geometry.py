#!/usr/bin/env python
#
# This code is part of Grandalf
# Copyright (C) 2008 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin, Fabio Zadrozny

from poset import *
from dot import *
import math
from math import atan,atan2,degrees,sqrt
from random import SystemRandom

try:
    from numpy import array,matrix,cos,sin,deg2rad
    has_numpy = True
except ImportError:
    has_numpy = False
    from math import cos,sin,pi
    deg2rad = lambda x: (x*math.pi/180.)
    from .linalg import array,matrix

#  rand_ortho1 returns a numpy.array representing
#  a random normalized n-dimension vector orthogonal to (1,1,1,...,1).
def  rand_ortho1(n):
    r = SystemRandom()
    pos = [r.random() for x in xrange(n)]
    s = sum(pos)
    v = array(pos,dtype=float)-(s/float(n))
    norm = sqrt(sum(v*v))
    return v/norm

#------------------------------------------------------------------------------
#TODO:  this was imported here from masr, but since we have
#  here access to numpy.array, we could use it for vectors operations.
def  intersect2lines((x1,y1),(x2,y2),(x3,y3),(x4,y4)):
    b = (x2-x1,y2-y1)
    d = (x4-x3,y4-y3)
    det = b[0]*d[1] - b[1]*d[0]
    if det==0: return None
    c = (x3-x1,y3-y1)
    t = float(c[0]*b[1] - c[1]*b[0])/(det*1.)
    if (t<0. or t>1.): return None
    t = float(c[0]*d[1] - c[1]*d[0])/(det*1.)
    if (t<0. or t>1.): return None
    x = x1 + t*b[0]
    y = y1 + t*b[1]
    return (x,y)


#------------------------------------------------------------------------------
#  intersectR returns the intersection point between the Rectangle
#  (w,h) that characterize the view object and the line that goes
#  from the views' object center to the 'topt' point.
def  intersectR(view,topt):
    # we compute intersection in local views' coord:
    # center of view is obviously :
    x1,y1 = 0,0
    # endpoint in view's coord:
    x2,y2 = topt[0]-view.xy[0],topt[1]-view.xy[1]
    # bounding box:
    bbx2 = view.w/2
    bbx1 = -bbx2
    bby2 = view.h/2
    bby1 = -bby2
    # all 4 segments of the bb:
    S = [((x1,y1),(x2,y2),(bbx1,bby1),(bbx2,bby1)),
              ((x1,y1),(x2,y2),(bbx2,bby1),(bbx2,bby2)),
              ((x1,y1),(x2,y2),(bbx1,bby2),(bbx2,bby2)),
              ((x1,y1),(x2,y2),(bbx1,bby2),(bbx1,bby1))]
    # check intersection with each seg:
    for segs in S:
        xy = intersect2lines(*segs)
        if xy!=None:
            x,y = xy
            # return global coord:
            x += view.xy[0]
            y += view.xy[1]
            return (x,y)
    # there can't be no intersection unless the endpoint was
    # inside the bb !
    raise ValueError('no intersection found (point inside ?!). view: %s topt: %s' % (view, topt))


#------------------------------------------------------------------------------
def  getangle(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    theta = atan2(y2-y1,x2-x1)
    return theta

#------------------------------------------------------------------------------
#  intersectC returns the intersection point between the Circle
#  of radius r and centered on views' position with the line
#  to the 'topt' point.
def  intersectC(view, r, topt):
    theta = getangle(view.xy,topt)
    x = int(cos(theta)*r)
    y = int(sin(theta)*r)
    return (x,y)


#------------------------------------------------------------------------------
def median_wh(views):
    mw = [v.w for v in views]
    mh = [v.h for v in views]
    mw.sort()
    mh.sort()
    return (mw[len(mw)/2],mh[len(mh)/2])

#------------------------------------------------------------------------------
#  setcurve returns the spline curve that path through the list of points P.
#  The spline curve is a list of cubic bezier curves (nurbs) that have
#  matching tangents at their extreme points.
#  The method considered here is taken from "The NURBS book" (Les A. Piegl,
#  Wayne Tiller, Springer, 1997) and implements a local interpolation rather
#  than a global interpolation.
def setcurve(e,pts,tgs=None):
    P = map(array,pts)
    n = len(P)
    # tangent estimation
    if tgs:
      assert len(tgs)==n
      T = map(array,tgs)
      Q = [ P[k+1]-P[k] for k in range(0,n-1)]
    else:
      Q,T = tangents(P,n)
    splines=[]
    for k in xrange(n-1):
        t = T[k]+T[k+1]
        a = 16. - (t.dot(t))
        b = 12.*(Q[k].dot(t))
        c = -36. * Q[k].dot(Q[k])
        D = (b*b) - 4.*a*c
        assert D>=0
        sd = sqrt(D)
        s1,s2 = (-b-sd)/(2.*a),(-b+sd)/(2.*a)
        s = s2
        if s1>=0: s=s1
        C0 = tuple(P[k])
        C1 = tuple(P[k] + (s/3.)*T[k])
        C2 = tuple(P[k+1] -(s/3.)*T[k+1])
        C3 = tuple(P[k+1])
        splines.append([C0,C1,C2,C3])
    return splines

#------------------------------------------------------------------------------
def tangents(P,n):
    assert n>=2
    Q = []
    T = []
    for k in xrange(0,n-1):
        q = P[k+1]-P[k]
        t = q/sqrt(q.dot(q))
        Q.append(q)
        T.append(t)
    T.append(t)
    return (Q,T)

#------------------------------------------------------------------------------
def setroundcorner(e,pts):
    P = map(array,pts)
    n = len(P)
    Q,T = tangents(P,n)
    c0 = P[0]
    t0 = T[0]
    k0 = 0
    splines = []
    k  = 1
    while k<n:
        z = abs(t0[0]*T[k][1]-(t0[1]*T[k][0]))
        if z<1.e-6:
            k+=1
            continue
        if (k-1)>k0: splines.append([c0,P[k-1]])
        if (k+1)<n:
            splines.extend(setcurve(e,[P[k-1],P[k+1]],tgs=[T[k-1],T[k+1]]))
        else:
            splines.extend(setcurve(e,[P[k-1],P[k]],tgs=[T[k-1],T[k]]))
            break
        if (k+2)<n:
            c0 = P[k+1]
            t0 = T[k+1]
            k0 = k+1
            k+=2
        else:
            break
    return splines or [[P[0],P[-1]]]

#------------------------------------------------------------------------------
class Point(object):
    '''
    Helper class representing a point.
    '''

    def __init__(self, *pts):
        self.x, self.y = pts

    def __getitem__(self, i):
        if i == 0:
            return self.x

        if i == 1:
            return self.y

        raise AssertionError('For 2d point can only get 0 or 1 (trying to get: %s)' % (i,))

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y

    def distance(self, p2):
        x1, y1 = self
        x2, y2 = p2
        dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def __str__(self):
        return '(%s, %s)' % (self.x, self.y)

    def __repr__(self):
        return 'Point(%s, %s)' % (self.x, self.y)

#------------------------------------------------------------------------------
def angle_between_360_degrees(angle):
    while angle >= 360.0:
        angle -= 360.0
    while angle < 0.0:
        angle += 360.0
    if angle == 360.0:
        angle = 0.0

    return angle

#------------------------------------------------------------------------------
def angle_to_x_axis_in_degrees(pt0, pt1):
    base_p0 = (0.0, 0.0)
    base_p1 = (1.0, 0.0)

    line1 = (pt0[0] - pt1[0], pt0[1] - pt1[1])
    line2 = (base_p1[0] - base_p0[0], base_p1[1] - base_p0[1])

    x1 = float(line1[0])
    y1 = float(line1[1])
    x2 = float(line2[0])
    y2 = float(line2[1])
    divide_by = (abs(x1 * x2) + abs(y1 * y2))

    if divide_by == 0.0:
        if pt0[1] > pt1[1]:
            return 90.0
        else:
            return 270.0
    else:
        tg = ((x1 * y2) - (x2 * y1)) / divide_by
        tan_degrees = degrees(atan(tg))

        if pt0[0] > pt1[0]:
            tan_degrees = -tan_degrees
        else:
            tan_degrees += 180.0

        angle = tan_degrees
        angle = angle_between_360_degrees(angle)

        return angle

#------------------------------------------------------------------------------
def new_point_at_distance(pt, distance, angle):
    angle = float(angle)
    x, y = pt[0], pt[1]
    x += float(distance) * cos(deg2rad(angle))
    y += float(distance) * sin(deg2rad(angle))
    return float(x), float(y)

