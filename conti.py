import networkx as nx
import numpy as np
import scipy 
import matplotlib.pylab as plt
import itertools
import random
import functools
from numpy import linalg as LA

"""
variation of delta deg and delta gap with lattice dimension 
"""

def remove_random_nodes(A,N):
    rs = random.sample(A.nodes, N)
    A.remove_nodes_from(rs)

def remove_random_links(A,E):
    rs = random.sample(A.edges, E)
    A.remove_edges_from(rs)
    
ddeg = []
dgap = []
rep = 10

N = np.linspace(0,200,num=25,dtype = int)

for i in N:
    ddi = []
    dgi = []
    for i in range(rep):
        R = nx.grid_graph(dim=[N,N]) 
        #M  = int(1/10*R.number_of_edges())
        #remove_random_links(R,M)
        A = nx.adj_matrix(R)
        L = nx.laplacian_matrix(R) 
        eigwd, eigv = LA.eigh(L.todense())
        eigw = np.sort(eigwd)
        eigw0 = eigw[0]
        eigw1 = eigw[1]
        eigw2 = eigw[2]
        eigw3 = eigw[3]
        ddi.append(np.abs((eigw1-eigw2))) 
        dgi.append(np.abs((eigw2-eigw3)))
        
    ddeg.append(np.mean(ddi))
    dgap.append(np.mean(dgi))

print(ddeg)
print(" ")
print(dgap)
print(N)

#%%

f = plt.figure(0)
a = f.subplots(1)
a.plot(N, ddeg,"coral")
a.plot(N, dgap, "crimson")
a.set_title("gap and degeration for unperturbed lattice with size N")
a.set_xlabel("lattice dimension")
a.set_ylabel("variation")
a.legend(("degenaration","gap"))


#%%

f = plt.figure(0)
g = plt.figure(1)
a = f.subplots(1)
b = g.subplots(1)

a.plot(N, ddeg,"coral")
b.plot(N, dgap, "crimson")
a.set_title("degeneration variation")
b.set_title("gap variation")
a.set_xlabel("lattice dimension")
a.set_ylabel("variation")
b.set_xlabel("lattice dimension")
b.set_ylabel("variation")
