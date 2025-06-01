# tests/test_cellular.py
import unittest
import os

from ga.cellular_cuda import CellularGA, PYCUDA_AVAILABLE
from ga.benchmarks import Sphere

@unittest.skipIf(not PYCUDA_AVAILABLE, "PyCUDA not available or CUDA context failed. Skipping CellularGA tests.")
class TestCellularGA(unittest.TestCase):

    def setUp(self):
        self.problem = Sphere(dimensions=3, target_fitness=1e-4, value_bounds=(-5,5))
        # CellularGA parameters for testing
        self.grid_width = 8
        self.grid_height = 8 # Population size = 64
        self.generations = 10 # Short run for testing
        self.crossover_rate = 0.9
        self.mutation_rate_gene = 0.05
        self.mutation_strength = 0.1
        self.neighborhood = "moore"

    def test_cellular_ga_initialization(self):
        """Test if CellularGA initializes without errors."""
        try:
            cga = CellularGA(
                problem=self.problem,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                generations=self.generations,
                crossover_rate=self.crossover_rate,
                mutation_rate_gene=self.mutation_rate_gene,
                mutation_strength=self.mutation_strength,
                neighborhood_type=self.neighborhood,
                verbose=False
            )
            self.assertIsNotNone(cga)
            self.assertEqual(cga.population_size, self.grid_width * self.grid_height)
            # Clean up GPU memory explicitly after test if CGA doesn't do it on del
            # (PyCUDA autoinit usually handles context, but explicit free in cga.run() is good)
        except Exception as e:
            self.fail(f"CellularGA initialization failed with PyCUDA available: {e}")


    def test_cellular_ga_runs_short_problem(self):
        """Test a short run of CellularGA."""
        cga = CellularGA(
            problem=self.problem,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            generations=self.generations, # Few generations
            crossover_rate=self.crossover_rate,
            mutation_rate_gene=self.mutation_rate_gene,
            mutation_strength=self.mutation_strength,
            neighborhood_type=self.neighborhood,
            verbose=False # Keep output minimal for tests
        )
        
        best_individual, best_fitness_log, avg_fitness_log = cga.run()

        self.assertIsNotNone(best_individual)
        self.assertTrue(best_individual.fitness < float('inf'))
        self.assertTrue(0 < len(best_fitness_log) <= self.generations) # Can terminate early
        self.assertTrue(0 < len(avg_fitness_log) <= self.generations)
        
        print(f"CellularGA ({self.grid_width}x{self.grid_height}) ran. Best fitness: {best_individual.fitness:.4f}")

    @unittest.skipIf(os.environ.get("CI", "false").lower() == "true", "Skipping longer CUDA test in CI if it might timeout or lack GPU.")
    def test_cellular_ga_von_neumann_neighborhood(self):
        """Test with Von Neumann neighborhood."""
        cga_vn = CellularGA(
            problem=self.problem,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            generations=5, # Very short
            crossover_rate=self.crossover_rate,
            mutation_rate_gene=self.mutation_rate_gene,
            mutation_strength=self.mutation_strength,
            neighborhood_type="von_neumann",
            verbose=False
        )
        best_ind_vn, _, _ = cga_vn.run()
        self.assertIsNotNone(best_ind_vn)
        self.assertTrue(best_ind_vn.fitness < float('inf'))
        print(f"CellularGA (Von Neumann) ran. Best fitness: {best_ind_vn.fitness:.4f}")

    # Add more tests:
    # - Test for different problem types if/when CUDA kernels are generalized.
    # - Test specific outcomes if RNG can be seeded/mocked for GPU (difficult).
    # - Test data transfers (e.g., check if initial population is correctly on GPU).
    # - Test error handling for invalid CUDA parameters or kernel failures (requires careful setup).

    # Note: Testing CUDA code thoroughly often requires a dedicated GPU in the test environment
    # and can be significantly more complex than CPU-bound code testing.
    # These tests are basic "smoke tests" to ensure it runs and produces some output.

if __name__ == '__main__':
    unittest.main()