# tests/test_master_slave.py
import unittest
import time
from multiprocessing import cpu_count

from ga.master_slave import MasterSlaveGA
from ga.base import TournamentSelection, ArithmeticCrossover, GaussianMutation
from ga.benchmarks import Sphere

class TestMasterSlaveGA(unittest.TestCase):

    def setUp(self):
        self.problem = Sphere(dimensions=5, target_fitness=1e-5)
        self.selection_op = TournamentSelection(tournament_size=3)
        self.crossover_op = ArithmeticCrossover(crossover_rate=0.8)
        self.mutation_op = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1, value_bounds=self.problem.value_bounds)
        
        self.pop_size = 50
        self.generations = 10 # Keep low for testing speed

    def test_master_slave_ga_runs_serial_like(self):
        """Test with 1 worker, should behave like serial."""
        ms_ga_serial = MasterSlaveGA(
            problem=self.problem,
            population_size=self.pop_size,
            generations=self.generations,
            selection_op=self.selection_op,
            crossover_op=self.crossover_op,
            mutation_op=self.mutation_op,
            elite_size=1,
            num_workers=1, # Explicitly serial
            verbose=False
        )
        start_time = time.time()
        best_ind, best_log, avg_log = ms_ga_serial.run()
        end_time = time.time()
        
        self.assertIsNotNone(best_ind)
        self.assertTrue(best_ind.fitness < float('inf'))
        self.assertEqual(len(best_log), self.generations) # Or until target met
        self.assertEqual(len(avg_log), self.generations) # Or until target met
        print(f"MasterSlaveGA (1 worker) ran in {end_time - start_time:.2f}s. Best fitness: {best_ind.fitness:.4f}")

    @unittest.skipIf(cpu_count() < 2, "Skipping parallel test: requires at least 2 CPU cores.")
    def test_master_slave_ga_runs_parallel(self):
        """Test with multiple workers."""
        num_parallel_workers = max(2, cpu_count() // 2) # Use at least 2, or half of available
        ms_ga_parallel = MasterSlaveGA(
            problem=self.problem,
            population_size=self.pop_size,
            generations=self.generations,
            selection_op=self.selection_op,
            crossover_op=self.crossover_op,
            mutation_op=self.mutation_op,
            elite_size=1,
            num_workers=num_parallel_workers,
            verbose=False
        )
        start_time = time.time()
        best_ind_p, best_log_p, avg_log_p = ms_ga_parallel.run()
        end_time = time.time()

        self.assertIsNotNone(best_ind_p)
        self.assertTrue(best_ind_p.fitness < float('inf'))
        # Length of logs can be less if target_fitness is met early
        self.assertTrue(len(best_log_p) <= self.generations)
        self.assertTrue(len(avg_log_p) <= self.generations)
        print(f"MasterSlaveGA ({num_parallel_workers} workers) ran in {end_time - start_time:.2f}s. Best fitness: {best_ind_p.fitness:.4f}")

        # Note: It's hard to assert speedup directly in a unit test due to overhead and variability,
        # but we can check it completes and produces a result.
        # A more rigorous test would compare runtimes over many iterations or larger problems.

    def test_master_slave_ga_handles_invalid_workers(self):
        """Test with 0 or negative workers, should default to 1 worker."""
        ms_ga_invalid_workers = MasterSlaveGA(
            problem=self.problem, population_size=self.pop_size, generations=5,
            selection_op=self.selection_op, crossover_op=self.crossover_op, mutation_op=self.mutation_op,
            num_workers=0, verbose=False
        )
        self.assertEqual(ms_ga_invalid_workers.num_workers, 1)
        
        ms_ga_neg_workers = MasterSlaveGA(
            problem=self.problem, population_size=self.pop_size, generations=5,
            selection_op=self.selection_op, crossover_op=self.crossover_op, mutation_op=self.mutation_op,
            num_workers=-2, verbose=False
        )
        self.assertEqual(ms_ga_neg_workers.num_workers, 1)


if __name__ == '__main__':
    unittest.main()