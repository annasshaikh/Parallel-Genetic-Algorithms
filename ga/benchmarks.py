# ga/benchmarks.py
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class BenchmarkProblem(ABC):
    def __init__(self, name: str, problem_type: str, dimensions: int = None, 
                 value_bounds: Tuple[float, float] = (-5.0, 5.0),
                 target_fitness: float = None):
        self.name = name
        self.problem_type = problem_type # "continuous", "combinatorial_permutation", "binary"
        self.dimensions = dimensions
        self.value_bounds = value_bounds # For continuous problems
        self.target_fitness = target_fitness # For termination condition
        self.evaluations = 0 # Counter for fitness evaluations

    @abstractmethod
    def evaluate(self, chromosome: Any) -> float:
        self.evaluations += 1 # Increment on each evaluation call
        pass

    @abstractmethod
    def generate_random_chromosome(self) -> Any:
        pass
    
    def get_chromosome_length(self) -> int:
        if self.dimensions:
            return self.dimensions
        # For problems where dimensions might not be the primary length descriptor (e.g., TSP num_cities)
        raise NotImplementedError("Chromosome length not explicitly defined for this problem via 'dimensions'. Override get_chromosome_length().")

    def is_optimal_solution(self, fitness: float) -> bool:
        if self.target_fitness is None:
            return False
        # Assuming minimization problems
        return fitness <= self.target_fitness
    
    def reset_evaluations(self):
        self.evaluations = 0

    def is_optimal_solution_known_for_logging(self): # Helper for CellularGA logging
        return self.target_fitness is not None


# --- Continuous Optimization Functions ---

class Sphere(BenchmarkProblem):
    def __init__(self, dimensions: int = 10, value_bounds: Tuple[float, float] = (-5.12, 5.12), target_fitness: float = 1e-6):
        super().__init__("Sphere", "continuous", dimensions, value_bounds, target_fitness)

    def evaluate(self, chromosome: List[float]) -> float:
        super().evaluate(chromosome) # Increment evaluation count
        if len(chromosome) != self.dimensions:
            raise ValueError(f"Chromosome length {len(chromosome)} does not match problem dimensions {self.dimensions}")
        return sum(x**2 for x in chromosome)

    def generate_random_chromosome(self) -> List[float]:
        low, high = self.value_bounds
        return [random.uniform(low, high) for _ in range(self.dimensions)]
    
    def get_chromosome_length(self) -> int:
        return self.dimensions

class Rastrigin(BenchmarkProblem):
    def __init__(self, dimensions: int = 10, value_bounds: Tuple[float, float] = (-5.12, 5.12), target_fitness: float = 1e-6):
        super().__init__("Rastrigin", "continuous", dimensions, value_bounds, target_fitness)
        self.A = 10.0

    def evaluate(self, chromosome: List[float]) -> float:
        super().evaluate(chromosome)
        if len(chromosome) != self.dimensions:
            raise ValueError(f"Chromosome length {len(chromosome)} does not match problem dimensions {self.dimensions}")
        return self.A * self.dimensions + sum([(x**2 - self.A * np.cos(2 * np.pi * x)) for x in chromosome])

    def generate_random_chromosome(self) -> List[float]:
        low, high = self.value_bounds
        return [random.uniform(low, high) for _ in range(self.dimensions)]

    def get_chromosome_length(self) -> int:
        return self.dimensions

class Griewank(BenchmarkProblem):
    def __init__(self, dimensions: int = 10, value_bounds: Tuple[float, float] = (-600.0, 600.0), target_fitness: float = 1e-3):
        super().__init__("Griewank", "continuous", dimensions, value_bounds, target_fitness)

    def evaluate(self, chromosome: List[float]) -> float:
        super().evaluate(chromosome)
        if len(chromosome) != self.dimensions:
            raise ValueError(f"Chromosome length {len(chromosome)} does not match problem dimensions {self.dimensions}")
        sum_sq = sum(x**2 / 4000.0 for x in chromosome)
        
        prod_cos = 1.0
        for i in range(len(chromosome)):
            prod_cos *= np.cos(chromosome[i] / np.sqrt(i + 1.0))
            
        return sum_sq - prod_cos + 1.0

    def generate_random_chromosome(self) -> List[float]:
        low, high = self.value_bounds
        return [random.uniform(low, high) for _ in range(self.dimensions)]

    def get_chromosome_length(self) -> int:
        return self.dimensions

# --- Combinatorial Problems ---

class TravelingSalesmanProblem(BenchmarkProblem):
    def __init__(self, name: str = "TSP", num_cities: int = 10, distance_matrix: np.ndarray = None, target_fitness: float = None):
        # Dimensions for TSP is num_cities
        super().__init__(name, "combinatorial_permutation", dimensions=num_cities, target_fitness=target_fitness)
        self.num_cities = num_cities
        if distance_matrix is None:
            # Generate random symmetric distance matrix (Euclidean)
            self.coords = np.random.rand(num_cities, 2) * 100 # Cities in a 100x100 area
            self.distance_matrix = self._calculate_distance_matrix(self.coords)
        else:
            if distance_matrix.shape != (num_cities, num_cities):
                raise ValueError("Distance matrix dimensions must match num_cities.")
            self.distance_matrix = distance_matrix
            self.coords = None 

    def _calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        num_cities = coords.shape[0]
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities): # Use i+1 for symmetric matrix, calculate once
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def evaluate(self, chromosome: List[int]) -> float: # Chromosome is a permutation of city indices
        super().evaluate(chromosome)
        if len(chromosome) != self.num_cities or len(set(chromosome)) != self.num_cities:
            # This check ensures it's a valid permutation of 0..N-1
            # A more robust check for specific range: if not all(0 <= x < self.num_cities for x in chromosome):
            return float('inf') # Penalize invalid solutions heavily

        total_distance = 0.0
        for i in range(self.num_cities):
            from_city = chromosome[i]
            to_city = chromosome[(i + 1) % self.num_cities] # Return to the starting city
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance

    def generate_random_chromosome(self) -> List[int]:
        # Generates a random permutation of city indices (0 to num_cities-1)
        chromosome = list(range(self.num_cities))
        random.shuffle(chromosome)
        return chromosome
    
    def get_chromosome_length(self) -> int:
        return self.num_cities

class RoyalRoad(BenchmarkProblem):
    """
    Royal Road R1 function (maximization, converted to minimization).
    A binary string is divided into k blocks of s bits.
    Fitness is sum of c_i * sigma_i(x) for each block i,
    where sigma_i(x) is 1 if all bits in block i are 1, and 0 otherwise.
    c_i is typically s (length of block). Optimal (max) fitness is num_blocks * block_size.
    We return (max_fitness - actual_fitness) for minimization.
    Target fitness for minimization is 0.
    """
    def __init__(self, block_size: int = 8, num_blocks: int = 4, target_fitness_minimization: float = 0.0):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.chromosome_length = block_size * num_blocks
        self.max_possible_fitness = self.num_blocks * self.block_size
        
        super().__init__("RoyalRoadR1", "binary", dimensions=self.chromosome_length, 
                         target_fitness=target_fitness_minimization) # Target for minimized version is 0

    def evaluate(self, chromosome: List[int]) -> float:
        super().evaluate(chromosome)
        if len(chromosome) != self.chromosome_length:
            raise ValueError(f"Chromosome length {len(chromosome)} does not match problem dimensions {self.chromosome_length}")

        actual_fitness = 0 # This is the sum to be maximized
        for i in range(self.num_blocks):
            start_index = i * self.block_size
            end_index = start_index + self.block_size
            block = chromosome[start_index:end_index]
            if all(bit == 1 for bit in block):
                actual_fitness += self.block_size # c_i = s (block_size)
        
        # Convert to minimization problem: Max Fitness - Actual Fitness
        # Optimal solution will have a fitness of 0.
        return self.max_possible_fitness - actual_fitness

    def generate_random_chromosome(self) -> List[int]:
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]

    def get_chromosome_length(self) -> int:
        return self.chromosome_length

    def is_optimal_solution(self, fitness: float) -> bool:
        # For minimization, target is 0
        return fitness <= self.target_fitness # target_fitness_minimization (e.g. 0)