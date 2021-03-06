import networkx as nx
import numpy as np
import scipy 
import pandas as pd
import matplotlib.pylab as plt
import itertools
import random
from scipy.sparse import csgraph as csgraph
from operator import itemgetter
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# creating lattice 
N = 15 # dimensioni
F = 0 # numero di nodi rimossi
L = 0 # numero di link rimossi

n_population = 100
elit_rate = 0.10
ELIT = int(n_population * elit_rate)
HALF = int(n_population / 2)
max_iter = 45
MUTATION_RATE = 0.25
precision = 1e-4

# to remove random link and nodes
def remove_random_nodes(A,N):
    rs = random.sample(A.nodes, N)
    A.remove_nodes_from(rs)

def remove_random_links(A,E):
    rs = random.sample(A.edges, E)
    A.remove_edges_from(rs)
    

def kabsch(pt_true, pt_guessed): # da rendere adatta a due d --> così lo è?
    # Translation 
    pt_true -= pt_true.mean(axis = 0)
    pt_guessed -= pt_guessed.mean(axis = 0)

    # Scaling
    pt_true /= np.linalg.norm(pt_true, axis=0)
    pt_guessed /= np.linalg.norm(pt_guessed, axis=0) # pt_guessed already normalized to 1 when eig results

    # find right permutation of axis
    combo = list(itertools.permutations(["x", "y"]))
    permutation = list(combo[ np.argmax( [ sum([ scipy.stats.pearsonr(x=pt_true[true], y=pt_guessed[guess])[0]
                                           for true, guess in zip(["x","y"], list(comb))
                                          ])
                                        for comb in combo ]
                                        )])
    V, S, W = np.linalg.svd( np.dot(pt_true.T, pt_guessed[permutation]) ) # SVD of covariance matrix
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    return pd.DataFrame(data = np.dot(pt_guessed[permutation], np.dot(V, W)), columns=["x", "y"]) # return pt_guessed rotated by U ( = V * W, rotation matrix)


#%%
    
def check_laplacian(m):
    lap = True 
    for i in range(N):
        for j in range(N):
            if m[i][j] != m[j][i]:
                lap = False 
                print("asymmetric")
    
    eigw, eigv = LA.eigh(m)   
    for k in range(1,N): 
        if eigw[k] < 0.: 
            lap = False 
            print("eigw prob", eigw[0], eigw[1], eigw[2], eigw[3])
        else: 
            continue
    return(lap)           

def get_cmap(coords, thr = 1.0):
    return scipy.sparse.csc_matrix(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords.iloc[:,1:], metric="euclidean") < thr), dtype=np.float)

def get_lap(cmap):
    return scipy.sparse.csc_matrix(csgraph.laplacian(cmap, normed=False))  # ma a questa davvero devo dare la cmap come argomento? non sono sicura per nulla
    
def fromcmaptocoords(C): 
    
    L = get_lap(C)
    eigw, eigv = LA.eigh(L.todense())
    vecs = eigv[:,1:3]
    guess_coords = pd.DataFrame(data=vecs, columns=["x", "y"])
    return guess_coords

def get_coord(lap):
    
    eigw, eigv = LA.eigh(lap)
    vecs = eigv[:,1:3]
    guess_coords = pd.DataFrame(data=vecs, columns=["x", "y"])
    return guess_coords

def random_population(lap, n_population, scale = 1e-2): 
    tri_lap = np.triu(lap.todense())
    rand_pop = []
    for i in range(n_population):
        masses = np.diagflat(np.random.uniform(low = 1.0, high = 2.0, size=(N*N)))
        smasses = np.sqrt(masses)
        inv_masses = np.linalg.inv(smasses)
        prd = np.dot(inv_masses, tri_lap) #prd = np.einsum('ij,jk->ik', inv_masses, tri_lap)
        rd = np.dot(prd, inv_masses) # rd = np.einsum('ij,jk->ik', prd, inv_masses)
        rrd = rd + rd.T #np.tril(rd) + np.tril(rd, -1).T
        for i in range(N*N):
            rrd[i][i] = rrd[i][i]/2. 
        rand_pop.append(rrd)
        if check_laplacian(rrd) == False:
            ew,ev = LA.eigh(rrd)
            print("Error in random pop")
    return rand_pop
        
    """
    rand_pop = [np.einsum('ij,jk->ik',np.diagflat(np.random.rand(1,N*N)),lap.todense()) for i in range(n_population)]
    return [w for w in rand_pop]
    """

def mutate(lap, wtype = "masses", mtype = "weak", scale=1e-2): 
    
    if mtype == "weak":
        trilap = np.triu(lap)
        nz = np.nonzero(trilap)
        index_a = random.choice(nz[0])
        index_b = random.choice(nz[1])
        trilap[index_a][index_b] = np.random.normal(loc=0.0, scale = scale, size=1)
        new_lap = trilap + trilap.T 
        for i in range(N*N):
            new_lap[i][i] = new_lap[i][i]/2.
    
    if mtype == "strong":
        trilap = np.triu(lap)
        ix = random.randrange(0,N,1)
        for j in range(N):
            if trilap[j][ix] != 0:
                trilap[j][ix] = np.random.normal(loc=0.0, scale = scale, size=1)[0]
        new_lap = trilap + trilap.T 
        for i in range(N*N):
            new_lap[i][i] = new_lap[i][i]/2.
    
    if check_laplacian(new_lap) == False:
        print("error in mutation")
    
    return new_lap

def crossover(lap_a, lap_b):
    
    trilap_a = np.triu(lap_a)
    trilap_b = np.triu(lap_b)
    for i in range(int(N/4),int(3*N/4),1):
        for j in range(int(N/4),int(3*N/4),1):
            trilap_a[i][j] = trilap_b[i][j]
    
    cross = trilap_a + trilap_a.T 
    for i in range(N*N):
        cross[i][i] = cross[i][i]/2.
    if check_laplacian(cross) == False:
        print("error in cross")
        
    return cross

def fitness(lap, t_coord): 
    
    gcoord =  get_coord(lap) 
    t_coord -= t_coord.mean(axis = 0)
    gcoord -= gcoord.mean(axis = 0)
    # Scaling
    t_coord /= np.linalg.norm(t_coord, axis=0)
    gcoord /= np.linalg.norm(gcoord, axis=0) 
    # find right permutation of axis
    combo = list(itertools.permutations(["x", "y"]))        
    permutation = list(combo[ np.argmax( [ sum([ scipy.stats.pearsonr(x=t_coord[true], y=gcoord[guess])[0]
                                                       for true, guess in zip(["x","y"], list(comb))
                                                      ])
                                                    for comb in combo ]
                                                    )])
    return np.sqrt( np.sum( np.sum( (t_coord - gcoord[permutation])**2, axis=1) )/N )

def new_generation(fit, idx, old_generation, real_coords, elit_rate = ELIT, mutation_rate = MUTATION_RATE, half = HALF, mtype = "strong"):
    #print("old_gen[0]",fitness(old_generation[0],real_coords))
    #fit = [fitness(individual, real_coords) for individual in old_generation]
    #sorted_gen = [x for y,x in sorted(zip(fit,old_generation))]
    new_gen =[]
    for i in range(n_population):
        if i < ELIT: 
            new_gen.append(old_generation[idx[i]])
        else: 
            new_gen.append(crossover(old_generation[np.random.randint(0,n_population/2.)], old_generation[np.random.randint(0,n_population/2.)]))
    
    for i in range(n_population):    
        if np.random.rand() < mutation_rate:
            new_gen[i] = mutate(new_gen[i],mtype)
    print("new gen")
    return new_gen

def GA(real_coords, laplacian, max_iter = max_iter):
   
    population = random_population(laplacian,n_population)
    
    for generation in range(max_iter):
        
        print("generation : %d"%generation)
        fit = [fitness(individual, real_coords) for individual in population]
        
        idx = np.argsort(fit) 
        best_id = idx[0]   
        print("best id",best_id)
        print("best fit",fit[best_id])
        
        if(fit[best_id] <= precision):
            break
        else:
            population = new_generation(fit, idx, population,real_coords)
            
    return population[best_id]


#%%
# creating lattice 
R = nx.grid_graph(dim=[N,N]) 
remove_random_nodes(R,F)
remove_random_links(R,L)

#%%
#true plot  
pos = np.asarray(R.nodes)
fig = plt.figure(0)
ax=fig.add_subplot(111,projection='3d')
ax.scatter(pos[:,0],pos[:,1], c='k',marker='o')
plt.show()

#%%
# creating adjacency matrix
A = nx.adj_matrix(R)
print("now printing adjacency matrix")
print(A.todense())

# creating laplacian matrix 
L = nx.laplacian_matrix(R)  # ritorna un numpy array
print("now printing laplacian matrix")
print(L.todense())

# computing eigenvalues and eigenvectors of the Laplacian Matrix 
eigwd, eigv = LA.eigh(L.todense())
eigw = np.sort(eigwd)

# taking only the second, third and fourth (the first one is zero)
ev1 = eigv[:,1]
ev2 = eigv[:,2]
vecs = eigv[:,1:3]
eigw0 = eigw[0]
eigw1 = eigw[1]
eigw2 = eigw[2]
eigw3 = eigw[3]
print("eigenvalues")
print(eigw0,eigw1,eigw2,eigw3)
print("\n")
print("creating dataframe")
_guess = pd.DataFrame(data=vecs, columns=["x", "y"])
print(_guess)

histo = plt.figure(1)
plt.hist(eigw,histtype="step",color="coral")
#%%
# kabsh 
tcoord = pd.DataFrame(np.asarray(R.nodes), columns=["x", "y"], dtype = float)
gcoord = kabsch(pd.DataFrame(np.asarray(R.nodes), columns=["x", "y"]), _guess ) 

# Translation
tcoord -= tcoord.mean(axis = 0)  
gcoord -= gcoord.mean(axis = 0)

# Scaling
tcoord /= np.linalg.norm(tcoord, axis=0)
gcoord /= np.linalg.norm(gcoord, axis=0) # gcoord already normalized to 1 when eig results

# find right permutation of axis
combo = list(itertools.permutations(["x", "y"]))        
permutation = list(combo[ np.argmax( [ sum([ scipy.stats.pearsonr(x=tcoord[true], y=gcoord[guess])[0]
                                                       for true, guess in zip(["x","y"], list(comb))
                                                      ])
                                                    for comb in combo ]
                                                    )])

tot_dist =  np.sqrt( np.sum( np.sum( (tcoord - gcoord[permutation])**2, axis=1) ) )
print("tot_dist is")
print(tot_dist)
RMSD = tot_dist
print("RMSD is")
print(np.sqrt( np.sum( np.sum( (tcoord - gcoord[permutation])**2, axis=1) )/N ))
#%%
# genetic algorithm
lap_genetic = GA(tcoord, L)

#%%
# plots
gc = get_coord(lap_genetic)

gcc = kabsch(pd.DataFrame(np.asarray(R.nodes), columns=["x", "y"]), gc) 

# Translation
tcoord -= tcoord.mean(axis = 0)  
gcc -= gcc.mean(axis = 0)

# Scaling
tcoord /= np.linalg.norm(tcoord, axis=0)
gcc /= np.linalg.norm(gcc, axis=0) # gcoord already normalized to 1 when eig results

# find right permutation of axis
combo = list(itertools.permutations(["x", "y"]))        
permutation = list(combo[ np.argmax( [ sum([ scipy.stats.pearsonr(x=tcoord[true], y=gcc[guess])[0]
                                                       for true, guess in zip(["x","y"], list(comb))
                                                      ])
                                                    for comb in combo ]
                                                    )])


fig = plt.figure(2)
bx=fig.add_subplot(111)
bx.plot(vecs[:,0], c="crimson")
bx.plot(vecs[:,1], c="coral")

fig = plt.figure(3)
cx=fig.add_subplot(111)
cx.plot(eigw, c="crimson")

fig = plt.figure(4)
dx=fig.add_subplot(111)
dx.scatter(np.arange(0,np.size(eigw),1),eigw, c="crimson", marker=".")
dx.set_xlim(0,15)
dx.set_ylim(0,0.11)
dx.set_title("")

# plot guessed 
fig = plt.figure(5)
ax=fig.add_subplot(111,projection='3d')
ax.scatter(_guess.x, _guess.y, linewidth= 1, c='k', marker='.')
ax.set_title("")

f = plt.figure(6)
ex = f.add_subplot(111)
ex.scatter(gc.x, gc.y, linewidth= 1, c='k', marker='*')




































 



