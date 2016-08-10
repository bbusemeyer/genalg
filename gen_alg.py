import numpy as np
import subprocess as sub
from copy import deepcopy

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
    self.best_solution = None
    self.population = None

  def optimize(self,init_pop, mutate_frac=0.1, nparents=2, 
      elitist_frac=0.1, max_gens=1000, fitness_goal=np.inf, nthread=8):
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
    for gen in range(max_gens):
      fitnesses = [self.fitness(unit) for unit in self.population]
      best = np.argmax(fitnesses)
      self.best_fitness = fitnesses[best]
      self.best_solution = self.population[best]
      if self.best_fitness > fitness_goal: break

      # Breed population by sampling best fitnesses.
      norm = np.linalg.norm(fitnesses,1)
      grabp = [fitness/norm for fitness in fitnesses]
      sel = np.random.choice(range(len(self.population)),
          nparents,replace=False,p=grabp)
      self.population = [ 
          self.breed([self.population[i] for i in sel]) 
          for unit in self.population
        ]
      
      #TODO mutate.
    if gen+1==max_gens: 
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

  def compute_greedy_solution(self,order=None):
    if type(order)==str:
      if order == 'sorted': 
        order = np.argsort(self.packages)[::-1]
      elif order=='backwards': 
        order = np.argsort(self.packages)
      else: 
        order = np.arange(len(self.packages))
        np.random.shuffle(order)
    else:
      order = np.arange(len(self.packages))
      np.random.shuffle(order)
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
