# ga/master_slave.py
import time
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Callable

from .base import BaseGA, Population, Individual, Selection, Crossover, Mutation
from .benchmarks import BenchmarkProblem

# Helper function for Pool.map - must be picklable (top-level or defined in a module)
# It takes a problem instance and a chromosome, returns chromosome and its fitness.

def evaluate_chromosome_task(args_tuple: Tuple[BenchmarkProblem, List[float]]) -> Tuple[List[float], float]:
    problem_instance, chromosome_to_eval = args_tuple
    fitness_value = problem_instance.evaluate(chromosome_to_eval)
    return chromosome_to_eval, fitness_value



class MasterSlaveGA(BaseGA):
    def __init__(self, problem: BenchmarkProblem, population_size: int, generations: int,
                 selection_op: Selection, crossover_op: Crossover, mutation_op: Mutation,
                 elite_size: int = 1, num_workers: int = None, verbose: bool = True):
        super().__init__(problem, population_size, generations, selection_op, crossover_op, mutation_op, elite_size, verbose)
        
        if num_workers is None:
            self.num_workers = cpu_count()
        elif num_workers <= 0:
            self.num_workers = 1 # Fallback to serial-like behavior if invalid num_workers
        else:
            self.num_workers = num_workers
        
        self.pool = None # Will be initialized in run()

        if self.verbose:
            logging.info(f"Master-Slave GA initialized with {self.num_workers} worker(s).")
            
    def _evaluate_population(self, population: Population):
        """
        Evaluates the fitness of individuals in the given Population object in parallel.
        Updates the fitness attribute of each Individual object in-place.
        This method is called for the initial population and for new offspring.
        """
        start_time = time.time()
        
        individuals_to_evaluate = [ind for ind in population.individuals if ind.fitness == float('inf')]
        
        if not individuals_to_evaluate:
            if self.verbose and self.current_generation % 20 == 0:
                 logging.debug(f"Gen {self.current_generation}: No new individuals to evaluate.")
            return # All individuals might have been evaluated (e.g., elites)

        chromosomes_only = [ind.chromosome for ind in individuals_to_evaluate]
        
        if self.pool and self.num_workers > 1 and chromosomes_only:
            # Prepare tasks: list of (problem_instance, chromosome) tuples
            # Passing self.problem directly. Each worker process gets a copy of it.
            tasks = [(self.problem, chromo) for chromo in chromosomes_only]
            
            # results_tuples is a list of (chromosome, fitness) in the same order as tasks
            results_tuples = self.pool.map(evaluate_chromosome_task, tasks)

            # Update fitness of the corresponding individuals
            for i, ind_obj in enumerate(individuals_to_evaluate):
                # original_chromosome_from_task, fitness_val = results_tuples[i]
                # We trust the order from pool.map.
                ind_obj.fitness = results_tuples[i][1] 
        elif chromosomes_only: # Fallback to serial evaluation if pool not active or num_workers=1
            if self.verbose and self.num_workers > 1 and not self.pool:
                logging.warning("MasterSlaveGA: Pool is not initialized, falling back to serial evaluation.")
            for i, ind_obj in enumerate(individuals_to_evaluate):
                # In serial case, evaluate_chromosome_task is not used, call problem.evaluate directly
                ind_obj.fitness = self.problem.evaluate(ind_obj.chromosome)
        
        end_time = time.time()
        if self.verbose and self.current_generation % 20 == 0 and individuals_to_evaluate:
             logging.debug(f"Gen {self.current_generation}: Evaluation of {len(individuals_to_evaluate)} individuals took {end_time - start_time:.4f}s")


    def run(self):
        # Initialize and manage the Pool within the run method
        if self.num_workers > 1:
            # 'with' statement ensures the pool is properly closed
            with Pool(processes=self.num_workers) as pool:
                self.pool = pool
                # Call the parent's run method which orchestrates the GA
                # The parent's run() will call self._evaluate_population() which now uses the pool.
                best_individual, best_fitness_log, avg_fitness_log = super().run()
            self.pool = None # Pool is closed and joined by 'with' context exit
        else:
            # Run serially if num_workers is 1 (no pool needed)
            self.pool = None 
            best_individual, best_fitness_log, avg_fitness_log = super().run()
            
        return best_individual, best_fitness_log, avg_fitness_log