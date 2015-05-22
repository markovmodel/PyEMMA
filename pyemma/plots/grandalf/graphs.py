#!/usr/bin/env python
#
# This code is part of Grandalf
# Copyright (C) 2008-2011 Axel Tillequin (bdcht3@gmail.com)
# published under GPLv2 license or EPLv1 license

from .utils import Poset
from operator import attrgetter

#------------------------------------------------------------------------------
#  vertex_core class:
#  e: list of edge_core objects.
#------------------------------------------------------------------------------
class  vertex_core(object):
    def __init__(self):
        # will hold list of edges for this vertex (adjacency list)
        self.e = []

    def deg(self): return len(self.e)

    def e_in(self):
        return filter( (lambda e:e.v[1]==self), self.e )

    def e_out(self):
        return filter( (lambda e:e.v[0]==self), self.e )

    def e_dir(self,dir):
        if dir>0: return self.e_out()
        if dir<0: return self.e_in()
        return self.e

    def N(self,f_io=0):
        N = []
        if f_io<=0: N += [ e.v[0] for e in self.e_in()  ]
        if f_io>=0: N += [ e.v[1] for e in self.e_out() ]
        return N

    def e_to(self,y):
        for e in self.e_out():
            if e.v[1]==y: return e
        return None

    def e_from(self,x):
        for e in self.e_in():
            if e.v[0]==x: return e
        return None

    def e_with(self,v):
        for e in self.e:
            if v in e.v: return e
        return None

    def detach(self):
        E = self.e[:]
        for e in E: e.detach()
        assert self.deg()==0
        return E


#------------------------------------------------------------------------------
#  edge_core class:
#  v = (x,y): a tuple of vertices (probably of type Vertex, see below)
#------------------------------------------------------------------------------
class  edge_core(object):
    def __init__(self,x,y):
        self.deg = 0 if x==y else 1
        self.v = (x,y)


#------------------------------------------------------------------------------
#  Vertex class enhancing a vertex_core with
#  c: the poset of connected vertices containing this vertex.
#  data: anything else associated with the vertex.
#------------------------------------------------------------------------------
class  Vertex(vertex_core):
    counter=0
    def __init__(self,data=None):
        vertex_core.__init__(self)
        self.index=Vertex.counter
        Vertex.counter += 1
        # by default, a new vertex belongs to its own component
        # but when the vertex is added to a graph, c points to the
        # connected component where it belongs.
        self.c = None
        self.data = data
    @classmethod
    def count(cls):
        return cls.counter
    def __hash__(self):
        return self.index

#------------------------------------------------------------------------------
#  Edge class:
#  w: weight associated with the edge, defaults to 1.
#  data: anything else associated with the edge.
#------------------------------------------------------------------------------
class  Edge(edge_core):
    counter=0
    def __init__(self,x,y,w=1,data=None,connect=False):
        edge_core.__init__(self,x,y)
        self.index=Edge.counter
        Edge.counter += 1
        # w is an optional weight associated with the edge.
        self.w = w
        self.data = data
        self.feedback = False
        if connect and (x.c==None or y.c==None):
            c = x.c or y.c
            c.add_edge(self)

    @classmethod
    def count(cls):
        return cls.counter
    def __hash__(self):
        return self.index

    def attach(self):
        if not self in self.v[0].e : self.v[0].e.append(self)
        if not self in self.v[1].e : self.v[1].e.append(self)

    def detach(self):
        if self.deg==1:
            assert self in self.v[0].e
            assert self in self.v[1].e
            self.v[0].e.remove(self)
            self.v[1].e.remove(self)
        else:
            if self in self.v[0].e: self.v[0].e.remove(self)
            assert self not in self.v[0].e
        return [self]


#------------------------------------------------------------------------------
#  graph_core class: A connected graph of Vertex/Edge objects.
#  self.sV: set of vertices
#  self.sE: set of edges
#  The graph is stored in edge list representation by self.sE,
#  but since the vertex_core embbeds edges information, the adjacency list rep
#  is straightforward from self.sV.
#------------------------------------------------------------------------------
class  graph_core(object):
    def __init__(self,V=None,E=None,directed=True):

        if V is None: V=[]
        if E is None: E=[]

        self.directed = directed

        self.sV = Poset(V)
        self.sE = Poset([])

        self.degenerated_edges=[]

        if len(self.sV)==1:
            v = self.sV[0]
            v.c = self
            for e in v.e: e.detach()
            return

        for e in E:
            x = self.sV.get(e.v[0])
            y = self.sV.get(e.v[1])
            if (x is None or y is None):
                raise ValueError,'unknown Vertex (%s or %s)'%e.v
            e.v = (x,y)
            if e.deg==0:
                e.detach()
                self.degenerated_edges.append(e)
                continue
            e = self.sE.add(e)
            e.attach()
            if x.c is None: x.c=Poset([x])
            if y.c is None: y.c=Poset([y])
            if id(x.c)!=id(y.c):
                x.c.update(y.c)
                y.c=x.c
            s=x.c
        #check if graph is connected:
        for v in self.V():
            if v.c is None or (v.c!=s):
                raise ValueError,'unconnected Vertex %s'%v.data
            else:
                v.c = self

    def roots(self):
        return filter(lambda v:len(v.e_in())==0, self.sV)

    def leaves(self):
        return filter(lambda v:len(v.e_out())==0, self.sV)

    # allow a graph_core to hold a single vertex:
    def add_single_vertex(self,v):
        if len(self.sE)==0 and len(self.sV)==0:
            v = self.sV.add(v)
            v.c = self
            return v
        return None

    # add edge e. At least one of its vertex must belong to the graph,
    # the other being added automatically.
    def add_edge(self,e):
        if e in self.sE:
            return self.sE.get(e)
        x = e.v[0]
        y = e.v[1]
        if not ((x in self.sV) or (y in self.sV)):
            raise ValueError,'unconnected edge'
        x = self.sV.add(x)
        y = self.sV.add(y)
        e.v = (x,y)
        e.attach()
        e = self.sE.add(e)
        x.c = self
        y.c = self
        return e

    # remove Edge :
    # this procedure checks that the resulting graph is connex.
    def remove_edge(self,e):
        if e.deg==0 or (not e in self.sE): return
        e.detach()
        # check if still connected (path is not oriented here):
        if not self.path(e.v[0],e.v[1]):
            # return to inital state by reconnecting everything:
            e.attach()
            # exit with exception!
            raise ValueError,e
        else:
            e = self.sE.remove(e)
            return e

    # remove Vertex:
    # this procedure checks that the resulting graph is connex.
    def remove_vertex(self,x):
        if x not in self.sV: return
        V = x.N() #get all neighbor vertices to check paths
        E = x.detach() #remove the edges from x and neighbors list
        # now we need to check if all neighbors are still connected,
        # and it is sufficient to check if one of them is connected to
        # all others:
        v0 = V.pop(0)
        for v in V:
            if not self.path(v0,v):
                # repair everything and raise exception if not connected:
                for e in E: e.attach()
                raise ValueError,x
        # remove edges and vertex from internal sets:
        for e in E: self.sE.remove(e)
        x = self.sV.remove(x)
        x.c = None
        return x

    # generates an iterator over vertices, with optional filter
    def V(self,cond=None):
        V = self.sV
        if cond is None: cond=(lambda x:True)
        for v in V:
            if cond(v):
                yield v

    # generates an iterator over edges, with optional filter
    def E(self,cond=None):
        E = self.sE
        if cond is None: cond=(lambda x:True)
        for e in E:
            if cond(e):
                yield e

    # vertex/edge properties :
    #-------------------------
    # returns number of vertices
    def order(self):
        return len(self.sV)

    # returns number of edges
    def norm(self):
        return len(self.sE)

    # returns the minimum degree
    def deg_min(self):
        return min([v.deg() for v in self.sV])

    # returns the maximum degree
    def deg_max(self):
        return max([v.deg() for v in self.sV])

    # returns the average degree d(G)
    def deg_avg(self):
        return sum([v.deg() for v in self.sV])/float(self.order())

    # returns the epsilon value (number of edges of G per vertex)
    def eps(self):
        return float(self.norm())/self.order()

    # shortest path between vertices x and y by breadth-first descent
    def path(self,x,y,f_io=0,hook=None):
        assert x in self.sV
        assert y in self.sV
        x = self.sV.get(x)
        y = self.sV.get(y)
        if x==y: return []
        if f_io!=0: assert self.directed==True
        # path:
        p = None
        if hook is None: hook = lambda x:False
        # apply hook:
        hook(x)
        # visisted:
        v = {x:None}
        # queue:
        q = [x]
        while (not p) and len(q)>0:
            c = q.pop(0)
            for n in c.N(f_io):
                if not v.has_key(n):
                    hook(n)
                    v[n] = c
                    if n==y: p = [n]
                    q.append(n)
                if p: break
        #now we fill the path p backward from y to x:
        while p and p[0]!=x:
            p.insert(0,v[p[0]])
        return p

    # shortest weighted-edges paths between x and all other vertices
    # by dijkstra's algorithm with heap used as priority queue.
    def dijkstra(self,x,f_io=0,hook=None):
        from collections import defaultdict
        from heapq import heappop, heappush
        if x not in self.sV: return None
        if f_io!=0: assert self.directed==True
        # initiate with path to itself...
        v = self.sV.get(x)
        # D is the returned vector of distances:
        D = defaultdict(lambda :None)
        D[v] = 0.0
        L = [(D[v],v)]
        while len(L)>0:
            l,u = heappop(L)
            for e in u.e_dir(f_io):
                v = e.v[0] if (u is e.v[1]) else e.v[1]
                Dv = l+e.w
                if D[v]!=None:
                    # check if heap/D needs updating:
                    # ignore if a shorter path was found already...
                    if Dv<D[v]:
                        for i,t in enumerate(L):
                            if t[1] is v:
                                L.pop(i)
                                break
                        D[v]=Dv
                        heappush(L,(Dv,v))
                else:
                    D[v]=Dv
                    heappush(L,(Dv,v))
        return D

    # returns the set of strongly connected components
    # ("scs") by using Tarjan algorithm.
    # These are maximal sets of vertices such that there is a path from each
    # vertex to every other vertex.
    # The algorithm performs a DFS from the provided list of root vertices.
    # A cycle is of course a strongly connected component,
    # but a strongly connected component can include several cycles.
    # The Feedback Acyclic Set of edge to be removed/reversed is provided by
    # marking the edges with a "feedback" flag.
    # Complexity is O(V+E).
    def get_scs_with_feedback(self,roots):
        from  sys import getrecursionlimit,setrecursionlimit
        limit=getrecursionlimit()
        N=self.norm()+10
        if N>limit:
            setrecursionlimit(N)
        def _visit(v,L):
            v.ind = v.ncur
            v.lowlink = v.ncur
            Vertex.ncur += 1
            self.tstack.append(v)
            v.mark = True
            for e in v.e_out():
                w = e.v[1]
                if w.ind==0:
                    _visit(w,L)
                    v.lowlink = min(v.lowlink,w.lowlink)
                elif w.mark:
                    e.feedback = True
                if w in self.tstack:
                    v.lowlink = min(v.lowlink,w.ind)
            if v.lowlink==v.ind:
                l=[self.tstack.pop()]
                while l[0]!=v:
                    l.insert(0,self.tstack.pop())
                #print "unstacked %s"%('-'.join([x.data[1:13] for x in l]))
                L.append(l)
            v.mark=False
        self.tstack=[]
        scs = []
        Vertex.ncur=1
        for v in self.sV: v.ind=0
        # start exploring tree from roots:
        for v in roots:
            v = self.sV.get(v)
            if v.ind==0: _visit(v,scs)
        # now possibly unvisited vertices:
        for v in self.sV:
            if v.ind==0: _visit(v,scs)
        # clean up Tarjan-specific data:
        for v in self.sV:
            del v.ind
            del v.lowlink
            del v.mark
        del Vertex.ncur
        del self.tstack
        setrecursionlimit(limit)
        return scs

    # returns neighbours of a vertex v:
    # f_io=-1 : parent nodes
    # f_io=+1 : child nodes
    # f_io= 0 : all (default)
    def N(self,v,f_io=0):
        return v.N(f_io)

    # general graph properties:
    # -------------------------

    # returns True iff
    #  - o is a subgraph of self, or
    #  - o is a vertex in self, or
    #  - o is an edge in self
    def __contains__(self,o):
        try:
            return o.sV.issubset(self.sV) and o.sE.issubset(self.sE)
        except AttributeError:
            return ((o in self.sV) or (o in self.sE))

    # merge graph_core G into self
    def union_update(self,G):
        for v in G.sV: v.c = self
        self.sV.update(G.sV)
        self.sE.update(G.sE)

    # derivated graphs:
    # -----------------

    # returns subgraph spanned by vertices V
    def spans(self,V):
        raise NotImplementedError

    # returns join of G (if disjoint)
    def __mul__(self,G):
        raise NotImplementedError

    # returns complement of a graph G
    def complement(self,G):
        raise NotImplementedError

    # contraction G\e
    def contract(self,e):
        raise NotImplementedError


#------------------------------------------------------------------------------
#  Graph class: Disjoint-set Graph.
#  V: list/set of vertices of type Vertex.
#  E: list/set of edges of type Edge.
#  The graph is stored in disjoint-sets holding each connex component
#  in self.C as a list of graph_core objects.
#------------------------------------------------------------------------------
class  Graph(object):
    component_class = graph_core

    def __init__(self,V=None,E=None,directed=True):
        if V is None: V=[]
        if E is None: E=[]
        self.directed = directed
        # tag connex set of vertices:
        # at first, every vertex is its own component
        for v in V: v.c = Poset([v])
        CV = [v.c for v in V]
        # then pass through edges and union associated vertices such that
        # CV finally holds only connected sets:
        for e in E:
            x = e.v[0]
            y = e.v[1]
            assert x in V
            assert y in V
            assert x.c in CV
            assert y.c in CV
            e.attach()
            if x.c!=y.c:
                #merge y.c into x.c :
                x.c.update(y.c)
                #update set list (MUST BE DONE BEFORE UPDATING REFS!)
                CV.remove(y.c)
                #update reference:
                for z in y.c: z.c = x.c
        # now create edge sets from connected vertex sets and
        # make the graph_core connected graphs for this component :
        self.C = []
        for c in CV:
            s = set()
            for v in c: s.update(v.e)
            self.C.append(self.component_class(c,s,directed))

    # add vertex v into the Graph as a new (unconnected) component
    def add_vertex(self,v):
        for c in self.C:
            if (v in c.sV): return c.sV.get(v)
        g = self.component_class(directed=self.directed)
        v = g.add_single_vertex(v)
        self.C.append(g)
        return v

    # add edge e and its vertices into the Graph possibly merging the
    # associated graph_core components
    def add_edge(self,e):
        # take vertices:
        x = e.v[0]
        y = e.v[1]
        x = self.add_vertex(x)
        y = self.add_vertex(y)
        # take respective graph_cores:
        cx = x.c
        cy = y.c
        # add edge:
        e = cy.add_edge(e)
        # connect (union) the graphs:
        if cx!=cy:
            cx.union_update(cy)
            self.C.remove(cy)
        return e

    def get_vertices_count(self):
        return sum([c.order() for c in self.C])

    # generates an iterator over vertices
    def V(self):
        for c in self.C:
            V = c.sV
            for v in V: yield v

    # generates an iterator over edges
    def E(self):
        for c in self.C:
            E = c.sE
            for e in E: yield e

    # remove Edge from a core
    def remove_edge(self,e):
        # get the graph_core:
        c = e.v[0].c
        assert c==e.v[1].c
        if not c in self.C: return None
        # remove edge in graph_core and replace it with two new cores
        # if removing edge disconnects the graph_core:
        try:
            e = c.remove_edge(e)
        except ValueError:
            e = c.sE.remove(e)
            e.detach()
            self.C.remove(c)
            tmpg = type(self)(c.sV,c.sE,self.directed)
            assert len(tmpg.C)==2
            self.C.extend(tmpg.C)
        return e

    # remove a Vertex and all its edges from a core
    def remove_vertex(self,x):
        # get the graph_core:
        c = x.c
        if not c in self.C: return None
        try:
            x = c.remove_vertex(x)
            if c.order()==0: self.C.remove(c)
        except ValueError:
            for e in x.detach(): c.sE.remove(e)
            x = c.sV.remove(x)
            self.C.remove(c)
            tmpg = type(self)(c.sV,c.sE,self.directed)
            assert len(tmpg.C)==2
            self.C.extend(tmpg.C)
        return x

    # vertex/edge properties :
    #-------------------------

    def order(self):
        return sum([c.order() for c in self.C])

    def norm(self):
        return sum([c.norm() for c in self.C])

    def deg_min(self):
        return min([c.deg_min() for c in self.C])

    def deg_max(self):
        return max([c.deg_max() for c in self.C])

    def deg_avg(self):
        t = 0.0
        for c in self.C: t += sum([v.deg() for v in c.sV])
        return t/float(self.order())

    def eps(self):
        return float(self.norm())/self.order()

    def path(self,x,y,f_io=0,hook=None):
        if x==y: return []
        if x.c!=y.c: return None
        # path:
        return x.c.path(x,y,f_io,hook)

    def N(self,v,f_io=0):
        return v.N(f_io)

    def __contains__(self,G):
        r = False
        for c in self.C: r |= (G in c)
        return r

    # returns True if Graph is connected
    def connected(self):
        return len(self.C)==1

    # returns connectivity (kappa)
    def connectivity(self):
        raise NotImplementedError

    # returns edge-connectivity (lambda)
    def e_connectivity(self):
        raise NotImplementedError

    # returns the list of graphs components
    def components(self):
        return self.C

    # derivated graphs:
    # -----------------

    # returns subgraph spanned by vertices V
    def spans(self,V):
        raise NotImplementedError

    # returns join of G (if disjoint)
    def __mul__(self,G):
        raise NotImplementedError

    # returns complement of a graph G
    def complement(self,G):
        raise NotImplementedError

    # contraction G\e
    def contract(self,e):
        raise NotImplementedError

