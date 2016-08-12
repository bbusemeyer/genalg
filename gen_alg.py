import numpy as np
from functools import partial
import time
import subprocess as sub
import multiprocessing as mp
import matplotlib.pyplot as plt
import plot_tools as pt
pt.matplotlib_header()
from copy import deepcopy

class GeneticOptimizer:
  def __init__(self,fitness,breed,nthreads=1):
    """ 
    fitness(sol) is a function that defines the fitness of a solution. Space of
    solutions is implicitly defined by fitness function. 
    breed(solist) takes a list of solutions and produces a new solution.
    mutate(sol,frac) takes a solution and perturbs a fraction of it. 
    """
    # FIXME Breed has issue when number of parents doesn't match what breed is expecting.
    self.fitness = fitness
    self.breed = breed
    self.best_fitness = -np.inf
    self.best_fitness_history = []
    self.best_solution = None
    self.population = None
    self.nthreads = nthreads


  def optimize(self,init_pop, mutate_frac=0.1, nparents=2, 
      elitist_frac=0.0, max_gens=1000, fitness_goal=np.inf, warn=False):
    """ 
    Perform genetic optimization on space defined by `self.fitness` and `init_pop`.
    mutate is the fraction of mutation for the gene.
    The top breeding_frac of population is bred. 
    Number of parents is parents. 
    Top elitist_frac of population is kept between generations.
    Keeps population constant, so breeding fraction is determined by number of
    parents.
    """
    self.population = init_pop
    keep = int(round(elitist_frac*len(self.population)))
    for gen in range(max_gens):
      fitnesses = [self.fitness(unit) for unit in self.population]
      #best_order = [np.argmax(fitnesses)]
      best_order = np.argsort(fitnesses)
      self.best_fitness = fitnesses[best_order[-1]]
      self.best_fitness_history.append(self.best_fitness)
      self.best_solution = self.population[best_order[-1]]
      if self.best_fitness > fitness_goal: break

      # Breed population by sampling best fitnesses.
      norm = np.linalg.norm(fitnesses,1)
      grabp = [fitness/norm for fitness in fitnesses]
      breed_pairs = [
          [
            self.population[packidx] for packidx in 
            np.random.choice(range(len(self.population)),nparents,replace=False,p=grabp)
          ]
          for p in self.population[keep:]
        ]
      self.population = [self.population[pidx] for pidx in best_order[-keep:]] +\
          [self.breed(pair) for pair in breed_pairs]
      #self.population = [self.best_solution] +\
      #    [self.breed(pair) for pair in breed_pairs]
      
      #TODO mutate.
    if gen+1==max_gens and warn: 
      print("Warning: did not reach fitness goal.")
    return self.best_fitness

class BinPackingProblem:
  def __init__(self,packages,binsize=1.0):
    self.packages = packages
    self.binsize = binsize

  def compute_fillings(self,packings):
    fillings = [0.0 for packing in packings]
    for packidx,packing in enumerate(packings):
      for pidx in packing:
        fillings[packidx] += self.packages[pidx]
    return fillings

  def greedy_pack(self,pidx,packings,fillings):
    for packidx,pack in enumerate(packings):
      if fillings[packidx]+self.packages[pidx] <= self.binsize:
        fillings[packidx] += self.packages[pidx]
        packings[packidx].append(pidx)
        return packings, fillings
    packings.append([pidx])
    fillings.append([self.packages[pidx]])
    return packings, fillings

  def compute_greedy_solution(self,order='unsorted'):
    if order == 'sorted': 
      order = np.argsort(self.packages)[::-1]
    elif order=='backwards': 
      order = np.argsort(self.packages)
    elif order=='unsorted':
      order = np.arange(len(self.packages))
      np.random.shuffle(order)
    else: 
      raise AssertionError("""
      Illegal order value: %s. 
      Available settings: 'sorted','backwards', or 'unsorted' (default).
      """%order)

    packings = [[]]
    fillings = [0.0]
    for pidx in order:
      packings, fillings = self.greedy_pack(pidx,packings,fillings)
    return packings

  def evaluate(self,packings):
    packed = [package for pack in packings for package in pack]
    packed = set(packed)
    fillings = self.compute_fillings(packings)
    if len(packed)!=len(self.packages):
      print("evaluate finds failed: some packages not packed.")
      return 0.0
    for filling in fillings:
      if filling>self.binsize:
        print("evaluate finds failed: some packs overfilled.")
        return 0.0
    return sum(fillings) / (self.binsize*len(packings))

class BinPackingGenAlg(BinPackingProblem):
  def breed_packings(self,packings_list):
    fillings_list = [self.compute_fillings(packings) for packings in packings_list]
    nremoved = min([len(pl) for pl in packings_list])
    premoved = [1.-fill for fill in fillings_list[0]]
    premoved /= sum(premoved)
    pcopy = [fill for fill in fillings_list[1]]
    pcopy /= sum(pcopy)
    removed  = np.random.choice(range(len(packings_list[0])),
        nremoved,p=premoved,replace=False)
    copyover = np.random.choice(range(len(packings_list[1])),
        nremoved,p=pcopy,replace=False)
    new_packings = deepcopy(packings_list[0])
    new_fillings = deepcopy(fillings_list[0])
    for remidx,packidx in enumerate(removed):
      new_packings[packidx] = packings_list[1][copyover[remidx]]
      new_fillings[packidx] = fillings_list[1][copyover[remidx]]
    ret = self._fix_packings(new_packings,new_fillings)[0]
    return ret

  def mutate_packings(self,packings_list):
    #TODO
    return packings_list

  def _fix_packings(self,packings,fillings):
    done = set()
    repack = []
    for packidx,packing in enumerate(packings):
      packset = set(packing)
      if len(done.intersection(packset)) > 0:
        packings.pop(packidx)
        fillings.pop(packidx)
        repack.extend(packset.difference(done))
      else:
        done = done.union(packset)
    repack.extend(set(range(len(self.packages))).difference(done))
    for pidx in repack:
      self.greedy_pack(pidx,packings,fillings)
    return packings, fillings

def greedy_hist(fig,ax,size,order='unsorted',stats=1000):
  res = []
  for i in range(stats):
    prob = BinPackingProblem(np.random.random(size))
    res.append(prob.evaluate(prob.compute_greedy_solution(order)))
  n,bins = np.histogram(res,normed=True)
  ax.bar(bins[:-1],n/n.sum(),width=bins[0]-bins[1])
  return fig,ax

def stat_greedy(order='sort',stats=300,size=100):
  ret = 0.0
  for i in range(stats):
    prob = BinPackingGenAlg(np.random.random(size))
    ret += prob.evaluate(prob.compute_greedy_solution(order))
  return ret/stats

def plot_size_trend(powers,stats=300,size=100):
  fig,ax = plt.subplots(1,1)
  res = []
  for i in [10*j for j in powers]:
    res.append((
        stat_greedy(order='sorted',   stats=stats,size=size),
        stat_greedy(order='unsorted',       stats=stats,size=size),
        stat_greedy(order='backwards',stats=stats,size=size)
      ))
  res = np.array(res).T
  ax.semilogx(10**powers,res[0],label='sort')
  ax.semilogx(10**powers,res[1],label='unsort')
  ax.semilogx(10**powers,res[2],label='backwards')
  ax.legend(loc='best',frameon=True)
  ax.set_xlabel("Number of packages")
  ax.set_ylabel("Fill fraction")
  return fig,ax

def test_genalg(max_gens,popsize,npackages,max_package=1.0):
	# Compare to random sampling of the same number of generations.
	prob = BinPackingGenAlg(np.random.random(npackages)*max_package)
	init_pop = [prob.compute_greedy_solution() for i in range(max_gens-2)]+\
			[prob.compute_greedy_solution('sorted'),prob.compute_greedy_solution('backwards')]
	init_fitnesses = [prob.evaluate(packings) for packings in init_pop]
	best_rand = max(init_fitnesses)
	print("Best random",best_rand)

	# Now optimize it with genetic algorithm.
	init_pop = [prob.compute_greedy_solution() for i in range(popsize-2)]+\
			[prob.compute_greedy_solution('sorted'),prob.compute_greedy_solution('backwards')]
	init_fitnesses = [prob.evaluate(packings) for packings in init_pop]
	optimizer = GeneticOptimizer(prob.evaluate,prob.breed_packings,nthreads=1)
	start=time.clock()
	optimizer.optimize(init_pop,elitist_frac=0.1,max_gens=100)
	end=time.clock()
	print("Best Vol. Frac.",optimizer.best_fitness)
	print("Time:",end-start)
	print("Generations",len(optimizer.best_fitness_history))
	plt.axhline(best_rand,color='k')
	plt.plot([dat for dat in optimizer.best_fitness_history])
	plt.ylabel("Volume fraction")
	plt.xlabel("Generation")

def _perform_genalg(dummy,max_gens,popsize,npackages,max_package=1.0):
  prob = BinPackingGenAlg(np.random.random(npackages)*max_package)
  init_pop = [prob.compute_greedy_solution() for i in range(popsize-2)]+\
      [prob.compute_greedy_solution('sorted'),prob.compute_greedy_solution('backwards')]
  init_fitnesses = [prob.evaluate(packings) for packings in init_pop]
  optimizer = GeneticOptimizer(prob.evaluate,prob.breed_packings,nthreads=1)
  start=time.clock()
  optimizer.optimize(init_pop,elitist_frac=0.1,max_gens=100)
  end=time.clock()
  return optimizer.best_fitness

def genalg_hist(fig,ax,stats,max_gens,popsize,npackages,max_package=1.0):
  onerun = partial(_perform_genalg, 
      max_gens=max_gens,
      popsize=popsize,
      npackages=npackages,
      max_package=max_package
    )
  with mp.Pool(8) as pool:
    res = pool.map(onerun,list(range(stats)))

  n,bins = np.histogram(res,normed=True)
  ax.bar(bins[:-1],n/n.sum(),width=bins[0]-bins[1])
  
def scan_max_package(fig,ax,stats,max_gens,popsize,npackages,max_packages):
  res = np.zeros(len(max_packages))
  for midx,max_package in enumerate(max_packages):
    print("Done with ",max_package)
    onerun = partial(_perform_genalg, 
        max_gens=max_gens,
        popsize=popsize,
        npackages=npackages,
        max_package=max_package
      )
    with mp.Pool(8) as pool:
      res[midx] = np.array(pool.map(onerun,list(range(stats)))).mean()
  ax.plot(res)
  
def scan_popsize(fig,ax,stats,max_gens,popsizes,npackages,max_package):
  res = np.zeros(len(popsizes))
  for midx,popsize in enumerate(popsizes):
    print("Done with ",popsize)
    onerun = partial(_perform_genalg, 
        max_gens=max_gens,
        popsize=popsize,
        npackages=npackages,
        max_package=max_package
      )
    with mp.Pool(8) as pool:
      res[midx] = np.array(pool.map(onerun,list(range(stats)))).mean()
  ax.plot(res)
