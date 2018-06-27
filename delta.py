import networkx as nx
import numpy as np
import matplotlib.pylab as plt
import random
from numpy import linalg as LA

"""
codice per ottenere gli andamenti di delta gap e delta deg normalizzati al variare del numero di link rimossi in maniera cumulativa 
"""

def remove_random_links(A,E):
    rs = random.sample(A.edges, E)
    A.remove_edges_from(rs)

N = 50
#LE = [0,60,30,40,30,30,20,40]
#LEC = [0,60,90,130,160,190,210,250]

LE = [0,50,50,50,50,50,50]
LEC = [0,50,150,200,250,300,350]

rep = 10
d = []
g = []

for j in range(rep):
    R = nx.grid_graph(dim=[N,N])  
    _deg = []
    _gap = [] 
    for i in LE:
        remove_random_links(R,i)
        A = nx.adj_matrix(R)
        L = nx.laplacian_matrix(R) 
        eigwd, eigv = LA.eigh(L.todense())
        eigw = np.sort(eigwd)
        eigw0 = eigw[0]
        eigw1 = eigw[1]
        eigw2 = eigw[2]
        eigw3 = eigw[3]
        deg = eigw2 - eigw1
        gap = eigw3 - eigw2
        _deg.append(deg)
        _gap.append(gap)
        print("meow")    
    d.append(_deg)
    g.append(_gap)
    print("M")
    

mean_deg = []
mean_gap = []
std_deg = []
std_gap = []

for j in range(np.size(LE)):
    mean_deg.append(np.mean([d[i][j] for i in range(rep)]))
    mean_gap.append(np.mean([g[i][j] for i in range(rep)]))
    std_deg.append(np.std([d[i][j] for i in range(rep)]))
    std_gap.append(np.std([g[i][j] for i in range(rep)]))



print(mean_deg)
print(" ")
print(mean_gap)
print(" ")
print(std_deg)
print(" ")
print(std_gap)

fig = plt.figure(0)
a = fig.subplots(1)
a.errorbar(LEC, mean_deg, yerr=std_deg, fmt=".", c = "coral")
a.set_xlabel("links removed")
a.set_ylabel("degeneration variation")

fig = plt.figure(1)
b = fig.subplots(1)
b.errorbar(LEC,mean_gap, yerr=std_gap, fmt=".", c = "crimson")
b.set_xlabel("links removed")
b.set_ylabel("gap variation")
