# tests/test_base.py
import unittest
import random
import numpy as np

# Adjust import path if your project structure requires it
# (e.g., if tests are outside the main package)
# from ..ga.base import Individual, Population, TournamentSelection, ArithmeticCrossover, GaussianMutation
# For simplicity, assuming tests are run from a context where 'ga' is directly importable
from ga.base import (Individual, Population, TournamentSelection, ArithmeticCrossover, 
                     TwoPointCrossover, OrderCrossoverTSP,
                     GaussianMutation, BitFlipMutation, SwapMutationTSP)
from ga.benchmarks import Sphere, TravelingSalesmanProblem, RoyalRoad


class TestIndividual(unittest.TestCase):
    def test_individual_creation_continuous(self):
        chromo = [1.0, 2.0, 3.0]
        ind = Individual(chromo, problem_type="continuous")
        self.assertEqual(ind.chromosome, chromo)
        self.assertEqual(ind.fitness, float('inf'))
        self.assertEqual(ind.problem_type, "continuous")

    def test_individual_creation_permutation(self):
        chromo = [0, 2, 1, 3]
        ind = Individual(chromo, problem_type="combinatorial_permutation")
        self.assertEqual(ind.chromosome, chromo)
        self.assertEqual(ind.problem_type, "combinatorial_permutation")

    def test_individual_clone_continuous(self):
        chromo = [1.5, -0.5]
        ind1 = Individual(list(chromo), problem_type="continuous") # Pass a copy
        ind1.fitness = 10.0
        
        ind2 = ind1.clone()
        self.assertNotEqual(id(ind1.chromosome), id(ind2.chromosome)) # Should be a deep copy for lists
        self.assertEqual(ind1.chromosome, ind2.chromosome)
        self.assertEqual(ind2.fitness, float('inf')) # Fitness is reset in clone typically
        self.assertEqual(ind1.problem_type, ind2.problem_type)
        
        # Test modification of cloned chromosome does not affect original
        ind2.chromosome[0] = 99.0
        self.assertNotEqual(ind1.chromosome[0], ind2.chromosome[0])
        self.assertEqual(ind1.chromosome[0], chromo[0])


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.ind1 = Individual([1,2], problem_type="continuous")
        self.ind1.fitness = 10.0
        self.ind2 = Individual([3,4], problem_type="continuous")
        self.ind2.fitness = 5.0 # Lower fitness is better (minimization)
        self.ind3 = Individual([5,6], problem_type="continuous")
        self.ind3.fitness = 15.0
        self.population = Population([self.ind1, self.ind2, self.ind3])

    def test_population_len(self):
        self.assertEqual(len(self.population), 3)

    def test_population_getitem(self):
        self.assertEqual(self.population[1].fitness, 5.0) # ind2

    def test_population_get_best_individual(self):
        best = self.population.get_best_individual()
        self.assertEqual(best.fitness, 5.0) # ind2 is best

    def test_population_get_best_fitness(self):
        self.assertEqual(self.population.get_best_fitness(), 5.0)

    def test_population_sort_by_fitness(self):
        self.population.sort_by_fitness()
        self.assertEqual(self.population.individuals[0].fitness, 5.0)  # ind2
        self.assertEqual(self.population.individuals[1].fitness, 10.0) # ind1
        self.assertEqual(self.population.individuals[2].fitness, 15.0) # ind3
    
    def test_empty_population(self):
        empty_pop = Population([])
        self.assertEqual(len(empty_pop), 0)
        self.assertIsNone(empty_pop.get_best_individual())
        self.assertEqual(empty_pop.get_best_fitness(), float('inf'))


class TestTournamentSelection(unittest.TestCase):
    def setUp(self):
        # Create a population with predictable fitness values
        self.individuals = []
        for i in range(10):
            # Fitness = 10, 9, 8, ..., 1 (best)
            ind = Individual([float(i)], problem_type="continuous")
            ind.fitness = 10.0 - i 
            self.individuals.append(ind)
        self.population = Population(self.individuals)
        random.seed(42) # For reproducible tests if random.sample is involved

    def test_tournament_selection_select_one(self):
        selector = TournamentSelection(tournament_size=3)
        # With a fixed seed and predictable fitness, we can somewhat predict outcomes
        # by manually running a few selections.
        # This is still stochastic, so exact individual isn't guaranteed without mocking random.sample
        winner = selector.select_one(self.population)
        self.assertIsInstance(winner, Individual)
        self.assertTrue(any(winner.chromosome == ind.chromosome and winner.fitness != float('inf') for ind in self.individuals))

    def test_tournament_selection_select_multiple(self):
        selector = TournamentSelection(tournament_size=3)
        num_selections = 5
        selected_parents = selector.select(self.population, num_selections)
        self.assertEqual(len(selected_parents), num_selections)
        for parent in selected_parents:
            self.assertIsInstance(parent, Individual)
            # Check if selected parent's chromosome came from original population
            # This is a bit tricky due to cloning. Check if a match exists.
            self.assertTrue(any(parent.chromosome == orig_ind.chromosome for orig_ind in self.individuals))
    
    def test_tournament_size_smaller_than_population(self):
        small_pop_individuals = [Individual([float(i)], problem_type="continuous") for i in range(2)]
        small_pop_individuals[0].fitness = 1.0
        small_pop_individuals[1].fitness = 2.0
        small_population = Population(small_pop_individuals)
        
        selector = TournamentSelection(tournament_size=3) # Tournament larger than pop
        winner = selector.select_one(small_population)
        self.assertEqual(winner.fitness, 1.0) # Should pick the best from the available

    def test_tournament_size_one(self):
        selector = TournamentSelection(tournament_size=1)
        selected_parents = selector.select(self.population, 3)
        self.assertEqual(len(selected_parents), 3)
        # Each selection just picks one random individual

    def test_empty_population_selection(self):
        empty_pop = Population([])
        selector = TournamentSelection(tournament_size=3)
        with self.assertRaises(ValueError): # select_one raises error
            selector.select_one(empty_pop)
        selected = selector.select(empty_pop, 5) # select method should handle gracefully
        self.assertEqual(len(selected), 0)


class TestCrossoverOperators(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.p1_cont = Individual([1.0, 2.0, 3.0, 4.0], problem_type="continuous")
        self.p2_cont = Individual([5.0, 6.0, 7.0, 8.0], problem_type="continuous")

        self.p1_bin = Individual([0,0,1,1,0,0], problem_type="binary")
        self.p2_bin = Individual([1,1,0,0,1,1], problem_type="binary")

        self.p1_tsp = Individual(list(range(6)), problem_type="combinatorial_permutation") # [0,1,2,3,4,5]
        random.shuffle(self.p1_tsp.chromosome) # e.g. [2, 4, 0, 1, 5, 3] with seed 42
        self.p2_tsp = Individual(list(range(6)), problem_type="combinatorial_permutation") # [0,1,2,3,4,5]
        random.shuffle(self.p2_tsp.chromosome) # e.g. [3, 1, 0, 5, 2, 4] with seed 42 (after first shuffle)

    def test_arithmetic_crossover(self):
        crossover_op = ArithmeticCrossover(crossover_rate=1.0) # Force crossover
        # Mock random.random() to return a fixed alpha for predictability
        original_random = random.random
        random.random = lambda: 0.5 # alpha = 0.5
        
        c1, c2 = crossover_op.perform_crossover(self.p1_cont, self.p2_cont)
        
        random.random = original_random # Restore original random
        
        expected_c1 = [0.5 * 1 + 0.5 * 5, 0.5 * 2 + 0.5 * 6, 0.5 * 3 + 0.5 * 7, 0.5 * 4 + 0.5 * 8] # [3,4,5,6]
        expected_c2 = [0.5 * 1 + 0.5 * 5, 0.5 * 2 + 0.5 * 6, 0.5 * 3 + 0.5 * 7, 0.5 * 4 + 0.5 * 8] # Also [3,4,5,6] because alpha=0.5
        self.assertEqual(c1.chromosome, expected_c1)
        self.assertEqual(c2.chromosome, expected_c2)
        self.assertEqual(c1.problem_type, "continuous")

    def test_two_point_crossover(self):
        crossover_op = TwoPointCrossover(crossover_rate=1.0)
        # Mock random.sample to return fixed points
        original_sample = random.sample
        random.sample = lambda population, k: [1, 4] # points p1=1, p2=4 (0-indexed) for len 6
        
        c1, c2 = crossover_op.perform_crossover(self.p1_bin, self.p2_bin)
        
        random.sample = original_sample # Restore
        
        # p1_bin: [0,  0,1,1,0,  0]
        # p2_bin: [1,  1,0,0,1,  1]
        # points 1, 4 means segment from index 1 up to (but not including) 4.
        # child1 = p1[0] + p2[1:4] + p1[4:] = [0] + [1,0,0] + [0,0] = [0,1,0,0,0,0]
        # child2 = p2[0] + p1[1:4] + p2[4:] = [1] + [0,1,1] + [1,1] = [1,0,1,1,1,1]
        # Correction: perform_crossover for TwoPoint in base.py uses p1, p2 (inclusive segment from p1, exclusive from p2 in example)
        # sorted(random.sample(range(chromosome_len), 2)) --> p1, p2 are indices
        # child1_chromo = parent1.chromosome[:p1] + parent2.chromosome[p1:p2] + parent1.chromosome[p2:]
        # If p1=1, p2=4 (meaning indices 1, 2, 3 from parent2)
        # p1_bin.chromosome = [0,0,1,1,0,0]
        # p2_bin.chromosome = [1,1,0,0,1,1]
        # c1_chromo = p1_bin[0:1] + p2_bin[1:4] + p1_bin[4:] = [0] + [1,0,0] + [0,0] = [0,1,0,0,0,0]
        # c2_chromo = p2_bin[0:1] + p1_bin[1:4] + p2_bin[4:] = [1] + [0,1,1] + [1,1] = [1,0,1,1,1,1]

        self.assertEqual(c1.chromosome, [0,1,0,0,0,0])
        self.assertEqual(c2.chromosome, [1,0,1,1,1,1])
        self.assertEqual(c1.problem_type, "binary")

    def test_order_crossover_tsp(self):
        crossover_op = OrderCrossoverTSP(crossover_rate=1.0)
        # For p1_tsp = [2, 4, 0, 1, 5, 3] and p2_tsp = [3, 1, 0, 5, 2, 4] (from seed 42)
        # Mock random.sample for crossover points
        original_sample = random.sample
        random.sample = lambda population, k: [2, 4] # start=2, end=4 (indices 2,3,4)
        
        c1, c2 = crossover_op.perform_crossover(self.p1_tsp, self.p2_tsp)
        random.sample = original_sample

        # p1: [2, 4, |0, 1, 5|, 3]
        # p2: [3, 1, |0, 5, 2|, 4]
        # Segment from p1 for c1: [0,1,5]
        # c1 starts as [-1, -1, 0, 1, 5, -1]
        # Remaining elements from p2 in order, skipping those in segment [0,1,5]:
        # p2 order after segment end (idx 4): p2[5]=4, p2[0]=3, p2[1]=1 (skip), p2[2]=0 (skip), p2[3]=5 (skip), p2[4]=2
        # Sequence from p2: [4, 3, 2]
        # Fill c1: c1[5]=4, c1[0]=3, c1[1]=2.
        # Expected c1: [3, 2, 0, 1, 5, 4]
        self.assertEqual(c1.chromosome, [3, 2, 0, 1, 5, 4])
        self.assertTrue(len(set(c1.chromosome)) == len(self.p1_tsp.chromosome)) # Check permutation

        # Segment from p2 for c2: [0,5,2]
        # c2 starts as [-1, -1, 0, 5, 2, -1]
        # Remaining elements from p1 in order:
        # p1 order after segment end (idx 4): p1[5]=3, p1[0]=2(skip), p1[1]=4, p1[2]=0(skip), p1[3]=1, p1[4]=5(skip)
        # Sequence from p1: [3, 4, 1]
        # Fill c2: c2[5]=3, c2[0]=4, c2[1]=1
        # Expected c2: [4, 1, 0, 5, 2, 3]
        self.assertEqual(c2.chromosome, [4, 1, 0, 5, 2, 3])
        self.assertTrue(len(set(c2.chromosome)) == len(self.p2_tsp.chromosome))

    def test_crossover_apply_method_rate(self):
        crossover_op = ArithmeticCrossover(crossover_rate=0.0) # Crossover never happens
        children = crossover_op.apply([self.p1_cont, self.p2_cont])
        self.assertEqual(children[0].chromosome, self.p1_cont.chromosome) # Clones of parents
        self.assertEqual(children[1].chromosome, self.p2_cont.chromosome)

        crossover_op_full = ArithmeticCrossover(crossover_rate=1.0) # Crossover always happens
        # Mock random.random for alpha in perform_crossover if needed for consistent output
        original_random = random.random
        random.random = lambda: 0.5 
        children_full = crossover_op_full.apply([self.p1_cont, self.p2_cont])
        random.random = original_random
        self.assertNotEqual(children_full[0].chromosome, self.p1_cont.chromosome) # Chromosomes should change
        
        # Test odd number of parents
        children_odd = crossover_op_full.apply([self.p1_cont, self.p2_cont, self.p1_bin.clone()])
        self.assertEqual(len(children_odd), 3)
        self.assertEqual(children_odd[2].chromosome, self.p1_bin.chromosome) # Last one cloned


class TestMutationOperators(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.ind_cont = Individual([1.0, 2.0, 3.0], problem_type="continuous")
        self.problem_cont = Sphere(dimensions=3, value_bounds=(-10,10)) # For bounds in Gaussian

        self.ind_bin = Individual([0,0,1,1,0], problem_type="binary")
        
        self.ind_tsp = Individual([0,1,2,3,4], problem_type="combinatorial_permutation")

    def test_gaussian_mutation(self):
        mut_op = GaussianMutation(mutation_rate=1.0, mutation_strength=0.1, value_bounds=(-10,10)) # Mutate every gene
        
        # Mock random.gauss for predictable mutation values
        original_gauss = random.gauss
        gauss_outputs = [0.05, -0.05, 0.1] # Cycle through these values
        gauss_idx = 0
        def mock_gauss(mu, sigma):
            nonlocal gauss_idx
            val = gauss_outputs[gauss_idx % len(gauss_outputs)]
            gauss_idx += 1
            return val # Sigma is strength, mu is 0
            
        random.gauss = mock_gauss
        # Also mock random.random for per-gene check (though rate is 1.0 here)
        original_random = random.random
        random.random = lambda: 0.0 # Ensures mutation happens

        mutated_ind = mut_op.perform_mutation(self.ind_cont.clone())
        
        random.gauss = original_gauss
        random.random = original_random

        expected_chromo = [
            1.0 + gauss_outputs[0], 
            2.0 + gauss_outputs[1], 
            3.0 + gauss_outputs[2]
        ]
        for i in range(len(expected_chromo)): # Clamp
             expected_chromo[i] = max(-10, min(10, expected_chromo[i]))

        self.assertEqual(mutated_ind.chromosome, expected_chromo)
        self.assertNotEqual(id(mutated_ind.chromosome), id(self.ind_cont.chromosome))

    def test_bit_flip_mutation(self):
        mut_op = BitFlipMutation(mutation_rate=1.0) # Flip every bit
        # Mock random.random for per-gene check (though rate is 1.0 here)
        original_random = random.random
        random.random = lambda: 0.0 # Ensures mutation happens
        
        mutated_ind = mut_op.perform_mutation(self.ind_bin.clone())
        
        random.random = original_random

        expected_chromo = [1,1,0,0,1] # All bits flipped from [0,0,1,1,0]
        self.assertEqual(mutated_ind.chromosome, expected_chromo)

    def test_swap_mutation_tsp(self):
        mut_op = SwapMutationTSP(mutation_rate=1.0) # Always attempt swap
        # Mock random.sample for predictable swap indices
        original_sample = random.sample
        random.sample = lambda population, k: [1, 3] # Swap index 1 and 3
        
        # Also mock random.random for per-individual check (though rate is 1.0 here)
        original_random = random.random
        random.random = lambda: 0.0 # Ensures mutation happens (per-individual check)

        mutated_ind = mut_op.perform_mutation(self.ind_tsp.clone())
        
        random.sample = original_sample
        random.random = original_random

        # Original: [0,1,2,3,4], swap idx 1 (value 1) and idx 3 (value 3)
        # Expected: [0,3,2,1,4]
        self.assertEqual(mutated_ind.chromosome, [0,3,2,1,4])

    def test_mutation_apply_method_rate(self):
        # Gaussian Mutation, rate 0.0 (never mutates per gene)
        mut_op_no_mutate_gene = GaussianMutation(mutation_rate=0.0, mutation_strength=0.1, value_bounds=(-10,10))
        cloned_ind_cont = self.ind_cont.clone()
        mutated_ind = mut_op_no_mutate_gene.apply(cloned_ind_cont) # apply calls perform_mutation on a clone
        self.assertEqual(mutated_ind.chromosome, self.ind_cont.chromosome) # No change to genes

        # Swap Mutation TSP, rate 0.0 (never mutates per individual)
        mut_op_no_mutate_individual = SwapMutationTSP(mutation_rate=0.0)
        cloned_ind_tsp = self.ind_tsp.clone()
        mutated_ind_tsp = mut_op_no_mutate_individual.apply(cloned_ind_tsp)
        self.assertEqual(mutated_ind_tsp.chromosome, self.ind_tsp.chromosome) # No change


if __name__ == '__main__':
    unittest.main()