# ga/base.py
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Tuple

class Individual:
    def __init__(self, chromosome: Any, problem_type: str = "continuous"):
        self.chromosome = chromosome
        self.fitness = float('inf') # Assuming minimization problems
        self.problem_type = problem_type # "continuous", "combinatorial_permutation", "binary"

    def __repr__(self):
        return f"Individual(chromosome={self.chromosome}, fitness={self.fitness})"

    def clone(self):
        # Ensure deep copy for mutable chromosomes like lists
        if isinstance(self.chromosome, list):
            cloned_chromosome = list(self.chromosome)
        elif isinstance(self.chromosome, np.ndarray):
            cloned_chromosome = np.array(self.chromosome)
        else:
            # For immutable types or types that handle their own copying
            cloned_chromosome = self.chromosome 
        
        cloned_ind = Individual(cloned_chromosome, self.problem_type)
        # Fitness is typically re-evaluated or set by GA, so default is fine for a fresh clone.
        # If fitness needs to be copied: cloned_ind.fitness = self.fitness
        return cloned_ind

class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, best_fitness={self.get_best_fitness()})"

    def get_best_individual(self) -> Individual:
        if not self.individuals:
            return None
        return min(self.individuals, key=lambda ind: ind.fitness)

    def get_best_fitness(self) -> float:
        if not self.individuals:
            return float('inf')
        best_ind = self.get_best_individual()
        return best_ind.fitness if best_ind else float('inf')

    def sort_by_fitness(self):
        self.individuals.sort(key=lambda ind: ind.fitness)

# --- Genetic Operators ---

class GAOperator(ABC):
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        pass

class Selection(GAOperator):
    @abstractmethod
    def select(self, population: Population, num_selections: int) -> List[Individual]:
        pass

class TournamentSelection(Selection):
    def __init__(self, tournament_size: int = 3):
        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1.")
        self.tournament_size = tournament_size

    def select_one(self, population: Population) -> Individual:
        if not population.individuals:
            raise ValueError("Cannot select from an empty population.")
        if len(population.individuals) < self.tournament_size:
            # If population is smaller than tournament size, pick from the whole population
            tournament_contenders = population.individuals
        else:
            tournament_contenders = random.sample(population.individuals, self.tournament_size)
        
        winner = min(tournament_contenders, key=lambda ind: ind.fitness)
        return winner.clone() # Return a copy

    def select(self, population: Population, num_selections: int) -> List[Individual]:
        selected = []
        if not population.individuals: # Cannot select if population is empty
            return selected
        for _ in range(num_selections):
            selected.append(self.select_one(population))
        return selected

    def apply(self, population: Population, num_selections: int) -> List[Individual]:
        return self.select(population, num_selections)


class Crossover(GAOperator):
    def __init__(self, crossover_rate: float):
        self.crossover_rate = crossover_rate

    @abstractmethod
    def perform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        pass

    def apply(self, parents: List[Individual]) -> List[Individual]:
        children = []
        # Ensure we have pairs of parents
        for i in range(0, len(parents) - (len(parents) % 2), 2):
            parent1, parent2 = parents[i], parents[i+1]
            if random.random() < self.crossover_rate:
                child1, child2 = self.perform_crossover(parent1, parent2)
                children.extend([child1, child2])
            else:
                # No crossover, parents pass through as clones
                children.extend([parent1.clone(), parent2.clone()])
        
        # If there's an odd parent out, it passes through cloned
        if len(parents) % 2 != 0:
            children.append(parents[-1].clone())
        return children

class ArithmeticCrossover(Crossover):
    """ For real-valued representations. """
    def perform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if parent1.problem_type != "continuous" or parent2.problem_type != "continuous":
            raise ValueError("Arithmetic crossover is for continuous problems.")
        if len(parent1.chromosome) != len(parent2.chromosome):
            raise ValueError("Parent chromosomes must have the same length for arithmetic crossover.")

        alpha = random.random() # Weight for crossover
        
        child1_chromo = [alpha * p1_gene + (1 - alpha) * p2_gene for p1_gene, p2_gene in zip(parent1.chromosome, parent2.chromosome)]
        child2_chromo = [(1 - alpha) * p1_gene + alpha * p2_gene for p1_gene, p2_gene in zip(parent1.chromosome, parent2.chromosome)]
        
        return Individual(child1_chromo, problem_type="continuous"), Individual(child2_chromo, problem_type="continuous")

class TwoPointCrossover(Crossover):
    """ For binary or list-based representations. """
    def perform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        chromosome_len = len(parent1.chromosome)
        if chromosome_len < 2: # Not enough length for two distinct points
            return parent1.clone(), parent2.clone() # Return clones of parents

        point1 = random.randint(0, chromosome_len - 2) 
        point2 = random.randint(point1 + 1, chromosome_len -1 )
        
        # Ensure points are distinct and in order, this logic is simplified.
        # A robust way: points = sorted(random.sample(range(chromosome_len), 2))
        # point1, point2 = points[0], points[1] (but range for sample needs to be at least 2)
        # Corrected point selection for two distinct points:
        if chromosome_len <= 1: # Cannot do two-point crossover
            return parent1.clone(), parent2.clone()
        
        p1, p2 = sorted(random.sample(range(chromosome_len), 2))


        child1_chromo = parent1.chromosome[:p1] + parent2.chromosome[p1:p2] + parent1.chromosome[p2:]
        child2_chromo = parent2.chromosome[:p1] + parent1.chromosome[p1:p2] + parent2.chromosome[p2:]
        
        return Individual(child1_chromo, problem_type=parent1.problem_type), Individual(child2_chromo, problem_type=parent2.problem_type)

class OrderCrossoverTSP(Crossover):
    """ Order Crossover (OX1) for TSP (permutation representations). """
    def perform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if parent1.problem_type != "combinatorial_permutation" or parent2.problem_type != "combinatorial_permutation":
            raise ValueError("Order Crossover is for permutation problems.")
        
        size = len(parent1.chromosome)
        if size == 0: return parent1.clone(), parent2.clone()

        child1_chromo, child2_chromo = [-1]*size, [-1]*size

        # Select two random crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Copy the segment from parent1 to child1 and parent2 to child2
        child1_chromo[start:end+1] = parent1.chromosome[start:end+1]
        child2_chromo[start:end+1] = parent2.chromosome[start:end+1]

        # Fill the remaining elements for child1
        # Elements from parent2 not in child1's copied segment, in order of appearance in parent2
        p2_idx = (end + 1) % size
        c1_idx = (end + 1) % size
        
        elements_from_p2 = []
        for i in range(size):
            current_p2_element = parent2.chromosome[( (end + 1 + i) % size)]
            if current_p2_element not in child1_chromo: # Check against already filled segment
                 elements_from_p2.append(current_p2_element)
        
        for val in elements_from_p2:
            if child1_chromo[c1_idx] == -1: # If spot is empty
                 child1_chromo[c1_idx] = val
            c1_idx = (c1_idx + 1) % size
            while child1_chromo[c1_idx] != -1 and -1 in child1_chromo: # Skip filled spots (segment)
                c1_idx = (c1_idx + 1) % size


        # Fill the remaining elements for child2
        p1_idx = (end + 1) % size
        c2_idx = (end + 1) % size
        elements_from_p1 = []
        for i in range(size):
            current_p1_element = parent1.chromosome[((end + 1 + i) % size)]
            if current_p1_element not in child2_chromo:
                elements_from_p1.append(current_p1_element)

        for val in elements_from_p1:
            if child2_chromo[c2_idx] == -1:
                child2_chromo[c2_idx] = val
            c2_idx = (c2_idx + 1) % size
            while child2_chromo[c2_idx] != -1 and -1 in child2_chromo:
                c2_idx = (c2_idx + 1) % size
            
        return Individual(child1_chromo, problem_type="combinatorial_permutation"), \
               Individual(child2_chromo, problem_type="combinatorial_permutation")


class Mutation(GAOperator):
    def __init__(self, mutation_rate: float): # This rate can be per-individual or per-gene depending on operator
        self.mutation_rate = mutation_rate

    @abstractmethod
    def perform_mutation(self, individual: Individual) -> Individual:
        pass

    def apply(self, individual: Individual) -> Individual:
        # The specific mutation logic (per-gene or per-individual check) is often inside perform_mutation
        return self.perform_mutation(individual.clone()) # Mutate a clone


class GaussianMutation(Mutation):
    """ For real-valued representations. Mutation rate is per-gene. """
    def __init__(self, mutation_rate: float, mutation_strength: float = 0.1, value_bounds: Tuple[float, float] = (-5.0, 5.0)):
        super().__init__(mutation_rate) # Per-gene rate
        self.mutation_strength = mutation_strength
        self.lower_bound, self.upper_bound = value_bounds

    def perform_mutation(self, individual: Individual) -> Individual: # individual is already a clone
        if individual.problem_type != "continuous":
            raise ValueError("Gaussian mutation is for continuous problems.")
        
        mutated_chromosome = individual.chromosome # Modify the clone's chromosome
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate: # Per-gene mutation check
                mutation_val = random.gauss(0, self.mutation_strength)
                mutated_chromosome[i] += mutation_val
                # Clamp to bounds
                mutated_chromosome[i] = max(self.lower_bound, min(self.upper_bound, mutated_chromosome[i]))
        # individual.chromosome is already updated
        return individual

class BitFlipMutation(Mutation):
    """ For binary representations. Mutation rate is per-gene. """
    def perform_mutation(self, individual: Individual) -> Individual: # individual is already a clone
        if individual.problem_type != "binary":
             raise ValueError("Bit-flip mutation is for binary problems.")
        mutated_chromosome = individual.chromosome
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate: # Per-gene mutation
                mutated_chromosome[i] = 1 - mutated_chromosome[i] # Flip bit
        return individual

class SwapMutationTSP(Mutation):
    """ For TSP (permutation representations). Mutation rate is per-individual. """
    def perform_mutation(self, individual: Individual) -> Individual: # individual is already a clone
        if individual.problem_type != "combinatorial_permutation":
            raise ValueError("Swap mutation is for permutation problems.")
        
        if random.random() < self.mutation_rate: # Per-individual mutation check
            mutated_chromosome = individual.chromosome
            size = len(mutated_chromosome)
            if size < 2: return individual # Cannot swap if less than 2 elements
            
            idx1, idx2 = random.sample(range(size), 2)
            mutated_chromosome[idx1], mutated_chromosome[idx2] = mutated_chromosome[idx2], mutated_chromosome[idx1]
        return individual

# --- Abstract Base Algorithm ---
class BaseGA(ABC):
    def __init__(self, problem, population_size: int, generations: int,
                 selection_op: Selection, crossover_op: Crossover, mutation_op: Mutation,
                 elite_size: int = 1,
                 verbose: bool = True):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.elite_size = elite_size if elite_size < population_size else 0
        self.verbose = verbose

        self.population: Population = None
        self.current_generation = 0
        self.best_fitness_over_time = []
        self.avg_fitness_over_time = []
        
    def _initialize_population(self):
        individuals = []
        for _ in range(self.population_size):
            chromosome = self.problem.generate_random_chromosome()
            individuals.append(Individual(chromosome, problem_type=self.problem.problem_type))
        self.population = Population(individuals)

    @abstractmethod
    def _evaluate_population(self, population: Population):
        """Evaluates individuals in the population, updates their fitness in-place."""
        pass

    def _reproduction(self) -> List[Individual]:
        # Selection
        # Number of individuals to generate through crossover/mutation
        num_offspring_to_generate = self.population_size - self.elite_size
        
        # Number of parents to select. Crossover typically takes 2 parents to make 2 children.
        # So, we need `num_offspring_to_generate` parents if crossover always produces 2 for 2.
        num_parents_to_select = num_offspring_to_generate
        
        parents = self.selection_op.select(self.population, num_parents_to_select)

        # Crossover
        children = self.crossover_op.apply(parents) # Produces roughly num_parents_to_select children
        
        # Mutation
        mutated_children = [self.mutation_op.apply(child) for child in children]
        
        return mutated_children # This list should ideally be of size num_offspring_to_generate

    def _replacement(self, offspring: List[Individual]):
        next_generation_individuals = []
        
        # Elitism: Keep the best individuals from the current population
        if self.elite_size > 0:
            self.population.sort_by_fitness() # Ensure current population is sorted for elitism
            elites = [self.population.individuals[i].clone() for i in range(self.elite_size)]
            next_generation_individuals.extend(elites)

        # Fill the rest of the population with new offspring
        needed = self.population_size - len(next_generation_individuals)
        next_generation_individuals.extend(offspring[:needed])
        
        # If, due to operator behavior, offspring count is less than needed,
        # this will result in a smaller population. Ensure pop size is maintained.
        if len(next_generation_individuals) < self.population_size:
            # This is a fallback, ideally offspring count matches `needed`.
            # Could fill with more from offspring (if available) or from old pop (non-elites)
            # For simplicity, we'll assume `_reproduction` and `_replacement` logic aligns
            # to produce/take `population_size - elite_size` individuals.
            # If offspring list is short, it means some parents were directly passed (cloned)
            # by crossover operator if crossover_rate was not 1.0.
            # The current Crossover.apply ensures output size is same as input parent list size.
            pass


        self.population = Population(next_generation_individuals)


    def run(self):
        self._initialize_population()
        self._evaluate_population(self.population) # Evaluate initial population (all individuals)

        for gen in range(self.generations):
            self.current_generation = gen
            
            self.population.sort_by_fitness() # Sort for logging and elitism
            best_ind_current_gen = self.population.get_best_individual()
            
            self.best_fitness_over_time.append(best_ind_current_gen.fitness)
            avg_fitness = sum(ind.fitness for ind in self.population.individuals) / len(self.population.individuals)
            self.avg_fitness_over_time.append(avg_fitness)

            if self.verbose and (gen % 10 == 0 or gen == self.generations - 1):
                print(f"Generation {gen}: Best Fitness = {best_ind_current_gen.fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")

            if self.problem.is_optimal_solution(best_ind_current_gen.fitness):
                if self.verbose:
                    print(f"Optimal or target solution found at generation {gen}.")
                break

            # Reproduction (Selection, Crossover, Mutation)
            offspring = self._reproduction() # Creates population_size - elite_size offspring
            
            # Evaluate ONLY the new offspring (elites already have fitness)
            # Create a temporary population of offspring to pass to _evaluate_population
            # This ensures that _evaluate_population (which might be parallel) only works on new individuals.
            offspring_population = Population(offspring)
            self._evaluate_population(offspring_population) # Evaluates fitness of individuals in 'offspring' list in-place
            
            # Replacement: Elites + evaluated offspring form the new population
            self._replacement(offspring_population.individuals) # Pass the list of evaluated offspring

            # The self.population is now the new generation (P_{t+1}).
            # All individuals in it (elites + evaluated offspring) have up-to-date fitness.
            # No need for another full _evaluate_population(self.population) call here if logic is tight.
            # The next iteration's sort_by_fitness will use these updated fitness values.

        self.population.sort_by_fitness() # Final sort
        final_best_individual = self.population.get_best_individual()
        if self.verbose:
            print(f"Finished GA run. Best individual: {final_best_individual}")
        return final_best_individual, self.best_fitness_over_time, self.avg_fitness_over_time