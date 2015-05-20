#!/usr/bin/env python

# This code is part of Grandalf
# Copyright (C) 2010-2012 Axel Tillequin (bdcht3@gmail.com)
# published under the GPLv2 license or EPLv1 license

#  Layouts are classes that provide graph drawing algorithms.
#
#  These classes all take a graph_core argument. The graph_core
#  topology will never be permanently modified by the drawing algorithm:
#  e.g. "dummy" node insertion, edge reversal for making the graph
#  acyclic and so on, are all kept inside the layout object.
#
from  bisect  import bisect
from  sys     import getrecursionlimit,setrecursionlimit

from grandalf.utils import *

try:
    xrange
except NameError:
    xrange = range

try:
    from itertools import izip
except:
    izip = zip

# the VertexViewer class is used as the default class
# for providing the Vertex dimensions (w,h) and position (xy)
# in its view attribute.
# The view object can however be instanciated from any ui widgets
# library as long as it provides the w,h,xy interface, allowing
# grandalf to get dimensions and set position directly from the widget.
class  VertexViewer(object):
    def __init__(self,w=2,h=2,data=None):
        self.w = w
        self.h = h
        self.data = data
        self.xy = None

    def __str__(self, *args, **kwargs):
        return 'VertexViewer (xy: %s) w: %s h: %s' % (self.xy, self.w, self.h)


#  SUGIYAMA LAYOUT
#------------------------------------------------------------------------------
#  The Sugiyama layout is the traditional "layered" graph layout
#  named 'dot' in graphviz.
#  This layout is quite efficient but heavily relies on drawing
#  heuristics. Adaptive drawing is limited to
#  extending the leaves only, but since the algorithm is quite fast
#  redrawing the entire graph (up to about a thousand nodes) gives
#  usually better results in less than a second.

class  _sugiyama_vertex_attr(object):
    def __init__(self,r=None,d=0):
        self.rank=r
        self.dummy=d
        self.pos=None
        self.x=0
        self.bar=None
    def __str__(self):
        s="(%3d,%3d) x=%s"%(self.rank,self.pos,str(self.x))
        if self.dummy: s="[d] %s"%s
        return s

class  DummyVertex(_sugiyama_vertex_attr):
    def __init__(self,r=None,viewclass=VertexViewer):
        self.view = viewclass()
        self.ctrl = None
        self.constrainer = False
        self.controlled = False
        _sugiyama_vertex_attr.__init__(self,r,d=1)
    def N(self,dir):
        assert dir==+1 or dir==-1
        return self.ctrl.get(self.rank+dir,[])
    def inner(self,dir):
        assert dir==+1 or dir==-1
        try:
            return self.ctrl[self.rank+dir][-1].dummy==1
        except KeyError:
            return False
        except AttributeError:
            return False
    def __str__(self):
        s="(%3d,%3d) x=%s"%(self.rank,self.pos,str(self.x))
        if self.dummy: s="[d] %s"%s
        if self.constrainer: s="[c] %s"%s
        return s

#------------------------------------------------------------------------------
# Layer is where Sugiyama layout organises nodes in hierarchical lists.
# The placement of nodes is done by the Sugiyama class, but it highly relies on
# the 'ordering' of nodes in each layer to reduce crossings.
# This ordering depends on the neighbors found in the upper or lower layers.
# WARNING: methods HIGHLY depend on layout.dirv state.
# TODO: see if grx dict can be stored inside Layers or needs to stay in layout.
#------------------------------------------------------------------------------
class Layer(list):
    __r    = None
    layout = None
    upper  = None
    lower  = None
    __x    = 1.
    ccount = None

    def __str__(self):
        s  = '<Layer %d'%self.__r
        s += ', len=%d'%len(self)
        xc = self.ccount or '?'
        s += ', crossings=%s>'%xc
        return s

    def setup(self,layout):
        self.layout = layout
        r = layout.layers.index(self)
        self.__r=r
        if len(self)>1: self.__x = 1./(len(self)-1)
        for i,v in enumerate(self):
            assert layout.grx[v].rank==r
            layout.grx[v].pos = i
            layout.grx[v].bar = i*self.__x
        if r>0:
            self.upper = layout.layers[r-1]
        if r<len(layout.layers)-1:
            self.lower = layout.layers[r+1]

    def nextlayer(self):
        return self.lower if self.layout.dirv==-1 else self.upper
    def prevlayer(self):
        return self.lower if self.layout.dirv==+1 else self.upper

    def order(self):
        sug = self.layout
        sug._edge_inverter()
        mvmt=[]
        c = self._cc()
        if c>0:
            for v in self:
                if sug.grx[v].dummy and v.constrainer:
                    mvmt.append(v)
                    continue
                bar = self._meanvalueattr(v,sug.order_attr)
                sug.grx[v].bar=bar
            while len(mvmt)>0:
                v = mvmt.pop()
                v.bar = sug.grx[v.ctrl[self.__r][0]].bar
            # now resort layers l according to bar value:
            self.sort(cmp=(lambda x,y: cmp(sug.grx[x].bar,sug.grx[y].bar)))
            # assign new position in layer l:
            for i,v in enumerate(self):
                if sug.grx[v].pos!=i: mvmt.append(v)
                sug.grx[v].pos = i
                #sug.grx[v].bar = i*self.__x
            # try count resulting crossings:
            c = self._ordering_reduce_crossings()
        self.layout._edge_inverter()
        self.ccount = c
        return mvmt

    # find new position of vertex v according to adjacency in prevlayer.
    # position is given by the mean value of adjacent positions.
    # experiments show that meanvalue heuristic performs better than median.
    def _meanvalueattr(self,v,att='bar'):
        sug = self.layout
        if sug.grx[v].dummy and v.constrainer:
            return v.ctrl[self.__r][0].bar
        if not self.prevlayer():
            return getattr(sug.grx[v],att)
        pos = [getattr(sug.grx[x],att) for x in self._neighbors(v)]
        if len(pos)==0:
            return getattr(sug.grx[v],att)
        return float(sum(pos))/len(pos)

    # find new position of vertex v according to adjacency in layer l+dir.
    # position is given by the median value of adjacent positions.
    # median heuristic is proven to achieve at most 3 times the minimum
    # of crossings (while barycenter achieve in theory the order of |V|)
    def _medianindex(self,v):
        assert self.prevlayer()!=None
        N = self._neighbors(v)
        g=self.layout.grx
        if g[v].dummy and v.controlled:
            for x in N:
                if g[x].dummy and x.constrainer:
                    return [g[x].pos]
        pos = [g[x].pos for x in N]
        lp = len(pos)
        if lp==0: return []
        pos.sort()
        pos = pos[::self.layout.dirh]
        i,j = divmod(lp-1,2)
        return [pos[i]] if j==0 else [pos[i],pos[i+j]]

    # neighbors refer to upper/lower adjacent nodes.
    # remember that v.N() provides neighbors of v within the graph, while
    # this method provides the Vertex and DummyVertex adjacent to v in the
    # upper or lower layer (depending on layout.dirv state).
    def _neighbors(self,v):
        assert self.layout.dag
        dirv = self.layout.dirv
        grxv=self.layout.grx[v]
        try: #(cache)
            return grxv.nvs[dirv]
        except AttributeError:
            grxv.nvs={-1:v.N(-1),+1:v.N(+1)}
            if grxv.dummy: return grxv.nvs[dirv]
            # v is real, v.N are graph neigbors but we need layers neighbors
            for d in (-1,+1):
                tr=grxv.rank+d
                for i,x in enumerate(v.N(d)):
                    if self.layout.grx[x].rank==tr:continue
                    e=v.e_with(x)
                    dum = self.layout.ctrls[e][tr][-1]
                    grxv.nvs[d][i]=dum
            return grxv.nvs[dirv]

    # counts (inefficently but at least accurately) the number of
    # crossing edges between layer l and l+dirv.
    # P[i][j] counts the number of crossings from j-th edge of vertex i.
    # The total count of crossings is the sum of flattened P:
    # x = sum(sum(P,[]))
    def _crossings(self):
        g=self.layout.grx
        P=[]
        for v in self:
            P.append([g[x].pos for x in self._neighbors(v)])
        for i,p in enumerate(P):
            candidates = sum(P[i+1:],[])
            for j,e in enumerate(p):
                p[j] = len(filter((lambda nx:nx<e), candidates))
            del candidates
        return P

    # implementation of the efficient bilayer cross counting by insert-sort
    # (see Barth & Mutzel paper "Simple and Efficient Bilayer Cross Counting")
    def _cc(self):
        g=self.layout.grx
        P=[]
        for v in self:
            P.extend(sorted([g[x].pos for x in self._neighbors(v)]))
        # count inversions in P:
        s = []
        count = 0
        for i,p in enumerate(P):
            j = bisect(s,p)
            if j<i: count += (i-j)
            s.insert(j,p)
        return count

    def _ordering_reduce_crossings(self):
        assert self.layout.dag
        g = self.layout.grx
        N = len(self)
        X=0
        for i,j in izip(xrange(N-1),xrange(1,N)):
            vi = self[i]
            vj = self[j]
            ni = [g[v].pos for v in self._neighbors(vi)]
            Xij=Xji=0
            for nj in [g[v].pos for v in self._neighbors(vj)]:
                x = len(filter((lambda nx:nx>nj),ni))
                Xij += x
                Xji += len(ni)-x
            if Xji<Xij:
                g[vi].pos,g[vj].pos = g[vj].pos,g[vi].pos
                self[i] = vj
                self[j] = vi
                X += Xji
            else:
                X += Xij
        return X

#------------------------------------------------------------------------------
#  The Sugiyama Layout Class takes as input a core_graph object and implements
#  an efficient drawing algorithm based on nodes dimensions provided through
#  a user-defined 'view' property in each vertex (see README.txt).
#------------------------------------------------------------------------------
class  SugiyamaLayout(object):
    def __init__(self,g):
        # drawing parameters:
        self.dirvh=0
        self.order_iter = 8
        self.order_attr = 'pos'
        self.xspace = 20
        self.yspace = 20
        self.dw = 10
        self.dh = 10
        # For layered graphs, vertices and edges need to have some additional
        # attributes that make sense only for this kind of layout:
        # update graph struct:
        self.g = g
        self.layers = []
        self.grx= {}
        self.ctrls = {}
        self.dag = False
        for v in self.g.V():
            assert hasattr(v,'view')
            self.grx[v] = _sugiyama_vertex_attr()
        self.dw,self.dh = median_wh([v.view for v in self.g.V()])
        self.dw = 8

    # initialize the layout engine based on required
    #  -list of edges for making the graph_core acyclic
    #  -list of root nodes.
    def init_all(self,roots=None,inverted_edges=None,cons=False,optimize=False):
        # For layered sugiyama algorithm, the input graph must be acyclic,
        # so we must provide a list of root nodes and a list of inverted edges.
        if roots==None:
            roots = filter(lambda x: len(x.e_in())==0, self.g.sV)
        if inverted_edges==None:
            L = self.g.get_scs_with_feedback(roots)
            inverted_edges = filter(lambda x:x.feedback, self.g.sE)
        self.alt_e = inverted_edges
        # assign rank to all vertices:
        self.rank_all(roots,optimize)
        # add dummy vertex/edge for 'long' edges:
        self.ctrls['cons']=cons  # use "constrained edges" ?
        for e in self.g.E():
            self.setdummies(e,cons)
        # precompute some layers values:
        for l in self.layers: l.setup(self)

    # compute every node coordinates after converging to optimal ordering by N
    # rounds, and finally perform the edge routing.
    def draw(self,N=1.5):
        while N>0.5:
            for (l,mvmt) in self.ordering_step():
                pass
            N = N-1
        if N>0:
            for (l,mvmt) in self.ordering_step(oneway=True):
                pass
        self.setxy()
        self.draw_edges()

    def _edge_inverter(self):
        for e in self.alt_e:
            x,y = e.v
            e.v = (y,x)
        self.dag = not self.dag

    # internal state for alignment policy:
    # dirvh=0 -> dirh=+1, dirv=-1: leftmost upper
    # dirvh=1 -> dirh=-1, dirv=-1: rightmost upper
    # dirvh=2 -> dirh=+1, dirv=+1: leftmost lower
    # dirvh=3 -> dirh=-1, dirv=+1: rightmost lower
    @property
    def dirvh(self): return self.__dirvh
    @property
    def dirv(self): return self.__dirv
    @property
    def dirh(self): return self.__dirh
    @dirvh.setter
    def dirvh(self,dirvh):
        assert dirvh in range(4)
        self.__dirvh=dirvh
        self.__dirh,self.__dirv={0:(1,-1), 1:(-1,-1), 2:(1,1), 3:(-1,1)}[dirvh]
    @dirv.setter
    def dirv(self,dirv):
        assert dirv in (-1,+1)
        dirvh = (dirv+1)+(1-self.__dirh)/2
        self.dirvh = dirvh
    @dirh.setter
    def dirh(self,dirh):
        assert dirh in (-1,+1)
        dirvh = (self.__dirv+1)+(1-dirh)/2
        self.dirvh = dirvh

    # rank all vertices.
    # if list l is None, find initial rankable vertices (roots),
    # otherwise update ranking from these vertices.
    # The initial rank is based on precedence relationships,
    # optimal ranking may be derived from network flow (simplex).
    def rank_all(self,roots,optimize=False):
        self._edge_inverter()
        r = filter(lambda x: len(x.e_in())==0 and x not in roots, self.g.sV)
        self._rank_init(roots+r)
        if optimize: self._rank_optimize()
        self._edge_inverter()

    def _rank_init(self,unranked):
        assert self.dag
        scan = {}
        # set rank of unranked based on its in-edges vertices ranks:
        while len(unranked)>0:
            l = []
            for v in unranked:
                self.setrank(v)
                # mark out-edges has scan-able:
                for e in v.e_out(): scan[e]=True
                # check if out-vertices are rank-able:
                for x in v.N(+1):
                    if not (False in [scan.get(e,False) for e in x.e_in()]):
                        if x not in l: l.append(x)
            unranked=l

    # TODO: Network flow solver minimizing total edge length
    # Also interesting: http://jgaa.info/accepted/2005/EiglspergerSiebenhallerKaufmann2005.9.3.pdf
    def _rank_optimize(self):
        assert self.dag
        for l in reversed(self.layers):
            for v in l:
                gv = self.grx[v]
                for x in v.N(-1):
                   if all((self.grx[y].rank>=gv.rank for y in x.N(+1))):
                        gx = self.grx[x]
                        self.layers[gx.rank].remove(x)
                        gx.rank = gv.rank-1
                        self.layers[gv.rank-1].append(x)


    def setrank(self,v):
        assert self.dag
        r=max([self.grx[x].rank for x in v.N(-1)]+[-1])+1
        self.grx[v].rank=r
        # add it to its layer:
        try:
            self.layers[r].append(v)
        except IndexError:
            assert r==len(self.layers)
            self.layers.append(Layer([v]))

    def dummyctrl(self,r,ctrl):
        dv = DummyVertex(r)
        dv.view.w,dv.view.h=self.dw,self.dh
        self.grx[dv] = dv
        dv.ctrl = ctrl
        try:
            ctrl[r].append(dv)
        except KeyError:
            ctrl[r] = [dv]
        self.layers[r].append(dv)
        return dv

    def setdummies(self,e,with_constraint=True):
        v0,v1 = e.v
        r0,r1 = self.grx[v0].rank,self.grx[v1].rank
        if r0>r1:
            assert e in self.alt_e
            v0,v1 = v1,v0
            r0,r1 = r1,r0
        elif r0==r1:
            raise ValueError,'bad ranking'
        spanover=xrange(r0+1,r1)
        if (r1-r0)>1:
            # "dummy vertices" are stored in the edge ctrl dict,
            # keyed by their rank in layers.
            ctrl=self.ctrls[e]={}
            ctrl[r0]=[v0]
            ctrl[r1]=[v1]
            for r in spanover:
                self.dummyctrl(r,ctrl)
            if e in self.alt_e and with_constraint:
                dv0 = self.dummyctrl(r0,ctrl)
                dv1 = self.dummyctrl(r1,ctrl)
                dv0.constrainer=True
                dv1.constrainer=True
                ctrl[r0+1][0].controlled=True
                ctrl[r1-1][0].controlled=True

    # iterator that computes all node coordinates and edge routing after
    # just one step (one layer after the other from top to bottom to top).
    # Purely inefficient ! Use it only for "animation" or debugging purpose.
    def draw_step(self):
        ostep = self.ordering_step()
        for s in ostep:
            self.setxy()
            self.draw_edges()
            yield s

    def ordering_step(self,oneway=False):
        self.dirv=-1
        crossings = 0
        for l in self.layers:
            mvmt = l.order()
            crossings += l.ccount
            yield (l,mvmt)
        if oneway or (crossings == 0):
            return
        self.dirv=+1
        while l:
            mvmt = l.order()
            yield (l,mvmt)
            l = l.nextlayer()

    # algorithm by Brandes & Kopf:
    def setxy(self):
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float('infinity')
        # initialize vertex coordinates attributes:
        for l in self.layers:
            for v in l:
                self.grx[v].root  = v
                self.grx[v].align = v
                self.grx[v].sink  = v
                self.grx[v].shift = inf
                self.grx[v].X     = None
                self.grx[v].x     = [0.0]*4
        curvh = self.dirvh # save current dirvh value
        for dirvh in xrange(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = curvh # restore it
        # vertical coordinate assigment of all nodes:
        Y = 0
        for l in self.layers:
            dY = max([v.view.h/2. for v in l])
            for v in l:
                vx = sorted(self.grx[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1]+vx[2])/2.
                # final xy-coordinates :
                v.view.xy = (avgm,Y+dY)
            Y += 2*dY+self.yspace
        self._edge_inverter()

    # mark conflicts between edges:
    # inner edges are edges between dummy nodes
    # type 0 is regular crossing regular (or sharing vertex)
    # type 1 is inner crossing regular (targeted crossings)
    # type 2 is inner crossing inner (avoided by reduce_crossings phase)
    def _detect_alignment_conflicts(self):
        curvh = self.dirvh # save current dirvh value
        self.dirvh=0
        self.conflicts = []
        for L in self.layers:
            last = len(L)-1
            prev = L.prevlayer()
            if not prev: continue
            k0=0
            k1_init=len(prev)-1
            l=0
            for l1,v in enumerate(L):
                if not self.grx[v].dummy: continue
                if l1==last or v.inner(-1):
                    k1=k1_init
                    if v.inner(-1):
                        k1=self.grx[v.N(-1)[-1]].pos
                    for vl in L[l:l1+1]:
                        for vk in L._neighbors(vl):
                            k = self.grx[vk].pos
                            if (k<k0 or k>k1):
                                self.conflicts.append((vk,vl))
                    l=l1+1
                    k0=k1
        self.dirvh = curvh # restore it

    # vertical alignment highly depends on dirh/dirv state.
    def _coord_vertical_alignment(self):
        dirh,dirv = self.dirh,self.dirv
        g = self.grx
        for l in self.layers[::-dirv]:
            if not l.prevlayer(): continue
            r=None
            for vk in l[::dirh]:
                for m in l._medianindex(vk):
                    # take the median node in dirv layer:
                    um = l.prevlayer()[m]
                    if g[um].dummy and um.controlled and not g[vk].dummy:
                        continue
                    # if vk is "free" align it with um's root
                    if g[vk].align is vk:
                        if dirv==1: vpair = (vk,um)
                        else:       vpair = (um,vk)
                        # if vk<->um link is used for alignment
                        if (vpair not in self.conflicts) and \
                           (r==None or dirh*r<dirh*m):
                            g[um].align = vk
                            g[vk].root = g[um].root
                            g[vk].align = g[vk].root
                            r = m


    def _coord_horizontal_compact(self):
        limit=getrecursionlimit()
        N=len(self.layers)+10
        if N>limit:
            setrecursionlimit(N)
        dirh,dirv = self.dirh,self.dirv
        g = self.grx
        L = self.layers[::-dirv]
        # recursive placement of blocks:
        for l in L:
            for v in l[::dirh]:
                if g[v].root is v:
                    self.__place_block(v)
        setrecursionlimit(limit)
        # mirror all nodes if right-aligned:
        if dirh==-1:
            for l in L:
                for v in l:
                    x = g[v].X
                    if x: g[v].X = -x
        # then assign x-coord of its root:
        inf=float('infinity')
        rb=inf
        for l in L:
            for v in l[::dirh]:
                g[v].x[self.dirvh] = g[g[v].root].X
                rs = g[g[v].root].sink
                s = g[rs].shift
                if s<inf:
                    g[v].x[self.dirvh] += dirh*s
                rb = min(rb,g[v].x[self.dirvh])
        # normalize to 0, and reinit root/align/sink/shift/X
        for l in self.layers:
            for v in l:
                #g[v].x[dirvh] -= rb
                g[v].root = g[v].align = g[v].sink = v
                g[v].shift = inf
                g[v].X = None

    # TODO: rewrite in iterative form to avoid recursion limit...
    def __place_block(self,v):
        g = self.grx
        if g[v].X==None:
            # every block is initially placed at x=0
            g[v].X = 0.0
            # place block in which v belongs:
            w = v
            while 1:
                j = g[w].pos-self.dirh # predecessor in rank must be placed
                r = g[w].rank
                if 0<= j <len(self.layers[r]):
                    wprec = self.layers[r][j]
                    delta = self.xspace+(wprec.view.w + w.view.w)/2.   # abs positive minimum displ.
                    # take root and place block:
                    u = g[wprec].root
                    self.__place_block(u)
                    # set sink as sink of prec-block root
                    if g[v].sink is v:
                        g[v].sink = g[u].sink
                    if g[v].sink<>g[u].sink:
                        s = g[u].sink
                        newshift = g[v].X-(g[u].X+delta)
                        g[s].shift = min(g[s].shift,newshift)
                    else:
                        g[v].X = max(g[v].X,(g[u].X+delta))
                # take next node to align in block:
                w = g[w].align
                # quit if self aligned
                if w is v: break

    # Basic edge routing applied only for edges with dummy points.
    # Enhanced edge routing can be performed by using the apropriate
    # route_with_ functions from routing.py in the edges' view.
    def draw_edges(self):
        for e in self.g.E():
            if hasattr(e,'view'):
                l=[]
                r0,r1 = None,None
                if self.ctrls.has_key(e):
                    D = self.ctrls[e]
                    r0,r1 = self.grx[e.v[0]].rank,self.grx[e.v[1]].rank
                    if r0<r1:
                        ranks = xrange(r0+1,r1)
                    else:
                        if self.ctrls['cons']:
                            ranks = xrange(r0,r1-1,-1)
                        else:
                            ranks = xrange(r0-1,r1,-1)
                    l = [D[r][-1].view.xy for r in ranks]
                l.insert(0,e.v[0].view.xy)
                l.append(e.v[1].view.xy)
                if self.ctrls['cons'] and r0>r1:
                    dy = e.v[0].view.h/2. + self.yspace/3.
                    x,y = zip(l[0],l[1])
                    y = max(y)+dy
                    l.insert(1,(x[0],y))
                    l.insert(2,(x[1],y))
                    dy = e.v[1].view.h/2. + self.yspace/3.
                    x,y = zip(l[-1],l[-2])
                    y = min(y)-dy
                    l.insert(-1,(x[0],y))
                    l.insert(-2,(x[1],y))
                try:
                    self.route_edge(e,l)
                except AttributeError:
                    pass
                e.view.setpath(l)


#  DIRECTED GRAPH WITH CONSTRAINTS LAYOUT
#------------------------------------------------------------------------------
class  DigcoLayout(object):
    def __init__(self, g, opt_x=True, opt_y=True):
        # drawing parameters:
        self.xspace = 10
        self.yspace = 10
        self.dr     = 10
        self.debug=False

        # initialize stress
        self.stress = float('inf')

	# set parameters
        self.opt_x = opt_x
        self.opt_y = opt_y
        self.g = g
	
        self.levels = []
        for i,v in enumerate(self.g.V()):
            assert hasattr(v,'view')
            v.i = i
            self.dr = max((self.dr,v.view.w,v.view.h))
        # solver parameters:
        self._cg_max_iter = g.order()
        self._cg_tolerance = 1.0e-6
        self._eps = 1.0e-5
        self._cv_max_iter = self._cg_max_iter

    def init_all(self,alpha=0.1,beta=0.01,x=None,y=None):
        if y is None:
            if self.g.directed:
                # partition g in hierarchical levels:
                y = self.part_to_levels(alpha,beta)
        # initiate positions (y and random in x):
        self.Z = self._xyinit(x=x, y=y)

    def draw(self,N=None):
        if N is None: N = self._cv_max_iter
        self.Z = self._optimize(self.Z,limit=N)
        # set view xy from near-optimal coords matrix:
        for v in self.g.V():
            v.view.xy = (self.Z[v.i][0,0]*self.dr,
                         self.Z[v.i][0,1]*self.dr)
        self.draw_edges()

    def draw_step(self):
        for x in xrange(self._cv_max_iter):
            self.draw(N=1)
            self.draw_edges()
            yield

    # Basic edge routing with segments
    def draw_edges(self):
        for e in self.g.E():
            if hasattr(e,'view'):
                l=[e.v[0].view.xy,e.v[1].view.xy]
                try:
                    self.route_edge(e,l)
                except AttributeError:
                    pass
                e.view.setpath(l)

    # partition the nodes into levels:
    def part_to_levels(self,alpha,beta):
        opty,err = self.optimal_arrangement()
        ordering = zip(opty,self.g.sV)
        eps = alpha*(opty.max()-opty.min())/(len(opty)-1)
        eps = max(beta,eps)
        ordering.sort(reverse=True)
        l = []
        self.levels.append(l)
        for i in xrange(len(ordering)-1):
            y,v = ordering[i]
            l.append(v)
            v.level = self.levels.index(l)
            if (y-ordering[i+1][0])>eps:
                l=[]
                self.levels.append(l)
        y,v = ordering[-1]
        l.append(v)
        v.level = self.levels.index(l)
        return opty

    def optimal_arrangement(self):
        b = self.balance()
        y = rand_ortho1(self.g.order())
        return self._conjugate_gradient_L(y,b)

    # balance vector is assembled in finite-element way...
    # this is faster than computing b[i] for each i.
    def balance(self):
        b = array([0.0]*self.g.order(),dtype=float)
        for e in self.g.E():
            s = e.v[0]
            d = e.v[1]
            q = e.w*(self.yspace+(s.view.h+d.view.h)/2.)
            b[s.i] += q
            b[d.i] -= q
        return b

    # We compute the solution Y of L.Y = b by conjugate gradient method
    # (L is semi-definite positive so Y is unique and convergence is O(n))
    # note that only arrays are involved here...
    def _conjugate_gradient_L(self,y,b):
        Lii = self.__Lii_()
        r = b - self.__L_pk(Lii,y)
        p = array(r,copy=True)
        rr = sum(r*r)
        for k in xrange(self._cg_max_iter):
            try:
                Lp = self.__L_pk(Lii,p)
                alpha = rr/sum(p*Lp)
                y += alpha/p
                r -= alpha*Lp
                newrr = sum(r*r)
                beta = newrr/rr
                rr = newrr
                if rr<self._cg_tolerance: break
                p = r + beta*p
            except ZeroDivisionError:
                return (None,rr)
        return (y,rr)

    # _xyinit can use diagonally scaled initial vertices positioning to provide
    # better convergence in constrained stress majorization
    def _xyinit(self, x=None, y=None):
        if x is None:
            x = rand_ortho1(self.g.order())
        if y is None:
            y = rand_ortho1(self.g.order())
        # translate and normalize:
        x = x - x[0]
        y = y - y[0]
        sfactor = 1.0 / max(map(abs,y)+map(abs,x))
        return matrix(zip(x*sfactor,y*sfactor))

    # provide the diagonal of the Laplacian matrix of g
    # the rest of L (sparse!) is already stored in every edges.
    def __Lii_(self):
        Lii = []
        for v in self.g.V():
            Lii.append(sum([e.w for e in v.e]))
        return array(Lii,dtype=float)

    # we don't compute the L.Pk matrix/vector product here since
    # L is sparse (order of |E| not |V|^2 !) so we let each edge
    # contribute to the resulting L.Pk vector in a FE assembly way...
    def __L_pk(self,Lii,pk):
        y = Lii*pk
        for e in self.g.sE:
            i1 = e.v[0].i
            i2 = e.v[1].i
            y[i1] -= e.w*pk[i2]
            y[i2] -= e.w*pk[i1]
        return y

    # conjugate_gradient with given matrix Lw:
    # it is assumed that b is not a multivector,
    # so _cg_Lw should be called in all directions separately.
    # note that everything is a matrix here, (arrays are row vectors only)
    def _cg_Lw(self,Lw,z,b):
        scal = lambda U,V: float(U.transpose()*V)
        r = b - Lw*z
        p = matrix(r,copy=True)
        rr = scal(r,r)
        for k in xrange(self._cg_max_iter):
            if rr<self._cg_tolerance: break
            Lp = Lw*p
            alpha = rr/scal(p,Lp)
            z = z + alpha*p
            r = r - alpha*Lp
            newrr = scal(r,r)
            beta = newrr/rr
            rr = newrr
            p  = r + beta*p
        return (z,rr)

    def __Dij_(self):
        Dji = []
        for v in self.g.V():
            wd = self.g.dijkstra(v)
            Di = [wd[w] for w in self.g.V()]
            Dji.append(Di)
        # at this point  D is stored by rows,
        # but anymway it's a symmetric matrix
        return matrix(Dji,dtype=float)

    # returns matrix -L^w
    def __Lij_w_(self):
        self.Dij = self.__Dij_()  # we keep D also for L^Z computations
        Lij = self.Dij.copy()
        n = self.g.order()
        for i in xrange(n):
            d = 0
            for j in xrange(n):
                if j==i: continue
                Lij[i,j] = 1.0/self.Dij[i,j]**2
                d += Lij[i,j]
            Lij[i,i] = -d
        return Lij

    # returns vector -L^Z.Z:
    def __Lij_Z_Z(self,Z):
        from math import sqrt
        scal = lambda U,V: float(U.transpose()*V)
        def dist(Zi,Zk):
            v = (Zi-Zk).transpose()
            return sqrt(scal(v,v))
        n = self.g.order()
        # init:
        lzz = Z.copy()*0.0 # lzz has dim Z (n x 2)
        liz = matrix([0.0]*n) # liz is a row of L^Z (size n)
        # compute lzz = L^Z.Z while assembling L^Z by row (liz):
        for i in xrange(n):
            iterk_except_i = (k for k in xrange(n) if k<>i)
            for k in iterk_except_i:
                liz[0,k] = 1.0/(self.Dij[i,k]*dist(Z[i],Z[k]))
            liz[0,i] = 0.0 # forced, otherwise next liz.sum() is wrong !
            liz[0,i] = -liz.sum()
            # now that we have the i-th row of L^Z, just dotprod with Z:
            lzz[i] = liz*Z
        return lzz

    def _optimize(self,Z,limit=100):
        scal = lambda U,V: float(U.transpose()*V)
        Lw = self.__Lij_w_()
        K = self.g.order()*(self.g.order()-1.0)/2.0
        count=0
        deep=0
        b  = self.__Lij_Z_Z(Z)
        # initialize x,y
        x = Z[1:,0]
        y = Z[1:,0]
        while count<limit:
            if self.debug:
                print "count %d"%count
                print "Z = ",Z
                print "b = ",b
            # find next Z by solving Lw.Z = b in every direction:
            if (self.opt_x):
                x,xerr = self._cg_Lw(Lw[1:,1:],Z[1:,0],b[1:,0])
            if (self.opt_y):
                y,yerr = self._cg_Lw(Lw[1:,1:],Z[1:,1],b[1:,1])
            Z[1:,0] = x
            Z[1:,1] = y
            if self.debug:
                print " cg -> "
                print Z,xerr,yerr
            # compute new stress:
            FZ = K-float(x.transpose()*b[1:,0] + y.transpose()*b[1:,1])
            # precompute new b:
            b  = self.__Lij_Z_Z(Z)
            # update new stress:
            FZ += 2*float(x.transpose()*b[1:,0] + y.transpose()*b[1:,1])
            # test convergence:
            if self.debug:
                print 'stress=%.10f'%FZ
            if self.stress==0.0 : break
            elif abs((self.stress-FZ)/self.stress)<self._eps:
                if deep==2:
                    break
                else:
                    deep += 1
            self.stress=FZ
            count += 1
        return Z


#------------------------------------------------------------------------------
class  DwyerLayout(object):
    def __init__(self):
        raise NotImplementedError

