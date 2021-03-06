from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import math
import numpy as np
import multiprocessing
from pathlib import Path


default_params = \
    {
        "cvt_samples": 25000,
        "batch_size": 100,
        "random_init": 1000,
        "sigma_iso": 0.01,
        "sigma_line": 0.2,
        "dump_period": 100,
        "parallel": True,
        "cvt_use_cache": True,
        "min": [0,0,-1,0,0,0],
        "max": [0.1,10,0,1,1,1,1],
        "sampled_mutation_rate": 0.05,
        "sampled_mutation": False,
        "sampled_mutation_array": []
    }


class Species:
    def __init__(self, x, desc, fitness):
        self.x = x
        self.desc = desc
        self.fitness = fitness
    
def scale(x,params):
    x_scaled = []
    for i in range(0,len(x)) :
        x_scaled.append(x[i] * (params["max"][i] - params["min"][i]) + params["min"][i])
    return np.array(x_scaled)

def variation(x, archive, params):
    y = x.copy()
    keys = list(archive.keys())
    z = archive[keys[np.random.randint(len(keys))]].x
    for i in range(0,len(y)):
        # iso mutation
        a = np.random.normal(0, (params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + a
        # line mutation
        b = np.random.normal(0, 20*(params["max"][i]-params["min"][i])/300.0, 1)
        y[i] =  y[i] + b*(x[i] - z[i])
    y_bounded = []
    for i in range(0,len(y)):
        elem_bounded = min(y[i],params["max"][i])
        elem_bounded = max(elem_bounded,params["min"][i])
        y_bounded.append(elem_bounded)
    return np.array(y_bounded)

def sampled_variation(x, archive, params):
    y = x.copy()
    for i in range(0,len(y)):
        if np.random.rand() < params["sampled_mutation_rate"]:
            y[i] =  np.random.choice(params["sampled_mutation_array"])
    # y_bounded = []
    # for i in range(0,len(y)):
    #     elem_bounded = min(y[i],params["max"][i])
    #     elem_bounded = max(elem_bounded,params["min"][i])
    #     y_bounded.append(elem_bounded)
    return np.array(y)


def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'


def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def __cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    if cvt_use_cache:
        fname = __centroids_filename(k, dim)
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1)
    k_means.fit(x)
    return k_means.cluster_centers_


def __make_hashable(array):
    return tuple(map(float, array))


# format: centroid fitness desc x \n
# centroid, desc and x are vectors
def __save_archive(archive, gen, name='archive_'):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = name + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")


def __add_to_archive(s, archive, kdt):
    niche_index = kdt.query([s.desc], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = __make_hashable(niche)
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
    else:
        archive[n] = s

# evaluate a single vector (x) with a function f and return a species
# t = vector, function


def evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return Species(z, desc, fit)

def validate(indiv):
    '''
    Gets the object of the species and validates
    whether to insert in the archive.
    '''
    return True

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f, n_niches=1000, n_gen=1000, params=default_params, file_prefix='archive_', initial_population=None, validate_indiv=validate):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT
    c = __cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    __write_centroids(c)

    # init archive (empty)
    archive = {}

    # main loop
    for g in range(0, n_gen + 1):
        to_evaluate = []
        if g == 0:  # random initialization
            if initial_population is None: #Generate random population
                print ("Initialization with random population")
                if params["sampled_mutation"]: #If sampled mutation is used
                    for i in range(0, params['random_init']):
                        x = np.random.choice(params["sampled_mutation_array"], dim_x)
                        to_evaluate += [(np.array(x), f)]
                else:    
                    for i in range(0, params['random_init']):
                        x = np.random.random(dim_x)
                        x = scale(x, params)
                        x_bounded = []
                        for j in range(0,len(x)):
                            elem_bounded = min(x[j],params["max"][j])
                            elem_bounded = max(elem_bounded,params["min"][j])
                            x_bounded.append(elem_bounded)
                        to_evaluate += [(np.array(x_bounded), f)]
            else:
                print ("Initialization with custom population")
                for x in initial_population: #Initialize with provided population
                    x_bounded = []
                    for j in range(0,len(x)):
                        elem_bounded = min(x[j],params["max"][j])
                        elem_bounded = max(elem_bounded,params["min"][j])
                        x_bounded.append(elem_bounded)
                    to_evaluate += [(np.array(x_bounded), f)]

        else:  # variation/selection loop
            keys = list(archive.keys())
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[np.random.randint(len(keys))]]
                # copy & add variation
                z = variation(x.x, archive, params) if not params["sampled_mutation"] else sampled_variation(x.x, archive, params)
                to_evaluate += [(z, f)]
        # parallel evaluation of the fitness
        if params['parallel'] == True:
            s_list = pool.map(evaluate, to_evaluate)
        else:
            s_list = map(evaluate, to_evaluate)
        # natural selection
        for s in s_list:
            if validate_indiv(s): # Measures to avoid unwanted individuals to give selective pressure
                __add_to_archive(s, archive, kdt)
        # write archive
        if g % params['dump_period'] == 0 and params['dump_period'] != -1:
            print("generation:", g)
            __save_archive(archive, g, name=file_prefix)
        __save_archive(archive, n_gen, name=file_prefix)
    return archive



# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])
    init = np.random.rand(100,6)
    archive = compute(2, 6, rastrigin, n_niches=5000, n_gen=2500, file_prefix='archive_map1_', initial_population=init)
