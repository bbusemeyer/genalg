import numpy as np
import subprocess as sub

class GeneticOptimizer:
  def __init__(self,fitness,breed):
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
    self.best_candidate = None
    self.population = None

  def optimize(self,init_pop, mutate_frac=0.1, nparents=2, 
      elitist_frac=0.1, max_gens=1000, fitness_tol=np.inf, nthread=8):
    """ 
    Perform genetic optimization on space defined by `self.fitness` and `init_pop`.
    mutate is the fraction of mutation for the gene.
    The top breeding_frac of population is bred. 
    Number of parents is parents. 
    Top elitist_frac of population is kept between generations.
    Keeps population constant, so breeding fraction is determined by number of
    parents.
    """
    self.best_fitness = max([self.fitness(unit) for unit in init_pop])
    self.population = init_pop
    for gen in range(max_gens):
      if self.best_fitness > fitness_tol: break

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

  def compute_greedy_solution(self,order=None):
    if order is None: order = np.argsort(self.packages)
    packings = [[]]
    fillings = [0.0]
    for pidx in order:
      packings, fillings = self.greedy_pack(pidx,packings,fillings)
    return packings

  def evaluate(self,packings):
    fillings = self.compute_fillings(packings)
    for filling in fillings:
      if filling>self.binsize:
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
    for remidx,packidx in enumerate(removed):
      packings_list[0][packidx] = packings_list[1][copyover[remidx]]
      fillings_list[0][packidx] = fillings_list[1][copyover[remidx]]
    return self._fix_packings(packings_list[0],fillings_list[0])

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
    for pidx in repack:
      self.greedy_pack(pidx,packings,fillings)
    return packings, fillings
