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

np.random.seed(123)

# creating lattice 
N = 20 # dimensioni
F = 0 # numero di nodi rimossi
L = 0 # numero di link rimossi

n_population = 100
elit_rate = 0.10
ELIT = int(n_population * elit_rate)
HALF = int(n_population / 2)
max_iter = 180
MUTATION_RATE = 0.5
precision = 1e-6
m = 150

# to remove random link and nodes
def remove_random_nodes(A,N):
    rs = random.sample(A.nodes, N)
    A.remove_nodes_from(rs)

def remove_random_links(A,E):
    rs = random.sample(A.edges, E)
    A.remove_edges_from(rs)
    
def kabsch(pt_true, pt_guessed): 
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


def get_coord(lap):
    eigw, eigv = LA.eig(lap)
    vecs = eigv[:,1:3]
    guess_coords = pd.DataFrame(data=vecs, columns=["x", "y"])
    
    return guess_coords

def random_population(dim = N*N, n_pop = n_population):
    # creo n_pop vettori di lunghezza N*N di masse random 
    #mu, sigma = 1.6, 0.6
    low, high = 0.0,1.0
    #diag = np.random.normal(mu,sigma,size=[n_pop,N*N])
    diag =  np.random.uniform(low,high,size=[n_pop,N*N])
    #diag = np.random.binomial(40,0.5,size=[n_pop,N*N])*0.1
    #diag = np.random.binomial(2,0.4,[n_pop,N*N])
    return diag 

def crossover(vec_a, vec_b):
    rd = np.random.randint(1,N*N)
    new_a = vec_a[0:rd]
    new_b = vec_b[rd:N*N]
    new = np.concatenate((new_a,new_b))

    return new

def mutate(vec):
    mu, sigma = 0.7, 0.1 # 0.7, 0.5 ||  2.1, 0.5  
    #low, high = 0.0,1.0
    for i in range(m):
        rd = np.random.randint(1,N*N)
        #new = np.random.uniform(low, high, 1)
        new = np.random.normal(mu, sigma, 1)
        vec[rd] = new
    
    return vec 

def new_generation(idx, old_generation, elit_rate = ELIT, mutation_rate = MUTATION_RATE, half = HALF):
    n_a = [old_generation[idx[i]] for i in range(0,ELIT)]
    n_b = [crossover(old_generation[idx[np.random.randint(0,n_population/2.)]], old_generation[idx[np.random.randint(n_population/2., n_population)]]) for j in range(ELIT, n_population)] 

    new_gen = np.concatenate((n_a,n_b))

    for j in range(1,n_population):
        if np.random.rand() < mutation_rate:
            new_gen[j] = mutate(new_gen[j])
    return new_gen


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
    
    return np.sqrt( np.sum( np.sum( (t_coord - gcoord[permutation])**2, axis=1) )/(N**2) )
    #return np.std(np.sum( (t_coord - gcoord[permutation])**2, axis=1) )


def GA(real_coords, laplacian, max_iter = max_iter):
    
    population = random_population()
    fittini = []
    
    for generation in range(max_iter): 
        print("generation : %d"%generation)
        fit = []
        for individual in population:
            d = np.diagflat(individual)
            dt = np.dot(d, laplacian.toarray())
            f = fitness(dt, real_coords)
            #print("fit calculated")
            fit.append(f)
        idx = np.argsort(fit) 
        print("idx:", idx)
        best_id = idx[0]   
        print("best id",best_id)
        print("best fit",fit[best_id])
        fittini.append(fit[best_id])
        
        if(fit[best_id] <= precision):
            break
        else:
            population = new_generation(idx, population)
    print("fittini:", fittini)
    plt.scatter(np.arange(0,max_iter), fittini)
    return population[0]



#%%
# creating lattice 
R = nx.grid_graph(dim=[N,N]) 
remove_random_nodes(R,F)
remove_random_links(R,L)

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
print(np.sqrt( np.sum( np.sum( (tcoord - gcoord[permutation])**2, axis=1) )/(N**2) ))
#%%
# genetic algorithm
tc = pd.DataFrame(np.asarray(R.nodes), columns=["x", "y"], dtype = float)
lap_genetic = GA(tc, L) 

#%%
masses = np.diag(LA.inv(np.diagflat(lap_genetic))) # così tiro fuori i valori delle masse 
#masses = lap_genetic
print("masses",masses)




nuovolaplaciano = np.dot(np.diagflat(lap_genetic), L.toarray())
print(nuovolaplaciano)
gc = get_coord(nuovolaplaciano)
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

#%%
# plot guessed 
fig = plt.figure(1)
ax=fig.add_subplot(111,projection='3d')
ax.scatter(gcoord.x, gcoord.y, linewidth= 1, c='coral', marker='*')
ax.scatter(tcoord.x, tcoord.y, linewidth= 1, c = 'dodgerblue', marker='.' )
ax.set_title("")
ax.legend(("guessed", "ideal"))

f = plt.figure(2)
ex = f.add_subplot(111,projection='3d')
ex.scatter(gcc.x, gcc.y, linewidth= 1, c = 'crimson', marker='*')
ex.scatter(tcoord.x, tcoord.y, linewidth= 1, c = 'dodgerblue', marker='.' )
ex.legend(("after mass tuning", "ideal"))

figs = plt.figure(3)
fx = figs.add_subplot(111,projection='3d')

fx.scatter(gcc.x, gcc.y, linewidth= 1, c = 'crimson', marker='*')
fx.scatter(tcoord.x, tcoord.y, linewidth= 1, c = 'dodgerblue', marker='.' )
fx.scatter(gcoord.x, gcoord.y, linewidth= 1, c = 'coral', marker='*')
fx.legend(( "after mass tuning","ideal", "guessed"))
#%%
ff = plt.figure(4)
bx = ff.add_subplot(111,projection='3d')
bx.scatter(gcc.x, gcc.y, c = masses, s = masses*100, marker = 'o')



#%%
import seaborn as sns
mx = np.split(masses,N)
mat = np.asmatrix(mx)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(mat, annot=True, linewidths=.5, ax=ax)

#%%




eigwals, eigvecs = LA.eigh(nuovolaplaciano)

fig = plt.figure(5)
cx=fig.add_subplot(111)
cx.scatter(np.arange(0,np.size(eigw),1), eigw, c="aquamarine", marker = ".", s = 18**2, alpha = 0.7)
cx.scatter(np.arange(0,np.size(eigwals),1), eigwals, c="crimson", marker = ".", s = 18**2, alpha = 0.7)
cx.set_xlim(0,15)
cx.set_ylim(0,0.11)
cx.legend(("original", "mass tuned"))






















