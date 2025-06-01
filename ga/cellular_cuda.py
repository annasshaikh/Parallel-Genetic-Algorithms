# ga/cellular_cuda.py
import time
import logging
import numpy as np
from .base import BaseGA, Population, Individual 
from .benchmarks import BenchmarkProblem

# Try to import PyCUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit # Initializes CUDA context on import
    from pycuda.compiler import SourceModule
    import pycuda.curandom as curandom # For GPU random number generation states
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    logging.warning("PyCUDA not found or CUDA context failed to initialize. CellularGA will not be available.")
    # Define dummy classes if PyCUDA is not available to prevent runtime errors on class definition
    class DummySourceModule:
        def get_function(self, name): return None
    SourceModule = DummySourceModule
    cuda = None 
    curandom = None


# --- CUDA Kernel Templates ---
# These are simplified and primarily for continuous problems (e.g., Sphere).
# Generalizing them for all benchmark types is a significant task.

# Fitness Evaluation Kernel (Example: Sphere function)
# This needs to be highly adaptable for different problems.
CUDA_FITNESS_KERNEL_TEMPLATE = """
__global__ void evaluate_fitness_kernel(
    const float* chromosomes_flat, // Input: Flattened array of all chromosomes
    float* fitness_values,         // Output: Array for fitness values
    int num_individuals,           // Total number of individuals (grid_width * grid_height)
    int chromosome_dim             // Dimension of each chromosome
    // TODO: Add problem-specific parameters here if needed (e.g., constants for Rastrigin)
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID

    if (idx < num_individuals) {{
        const float* individual_chromosome = chromosomes_flat + idx * chromosome_dim;
        float fitness = 0.0f;

        // --- START Problem-Specific Fitness Calculation (Example: Sphere) ---
        for (int i = 0; i < chromosome_dim; ++i) {{
            float gene = individual_chromosome[i];
            fitness += gene * gene; // Sphere: sum(x_i^2)
        }}
        // --- END Problem-Specific Fitness Calculation ---
        
        fitness_values[idx] = fitness;
    }}
}}
"""

# Interaction Kernel (Selection, Crossover, Mutation within Neighborhoods)
CUDA_INTERACTION_KERNEL_TEMPLATE = """
#include <curand_kernel.h> // For cuRAND states and functions

// --- START Helper __device__ functions for genetic operators ---
// These mirror logic from base.py but in CUDA C for GPU execution.

// Example: Arithmetic Crossover for continuous chromosomes
__device__ void arithmetic_crossover_gpu(
    const float* p1_chromo, const float* p2_chromo, // Parent chromosomes
    float* offspring_chromo,                        // Output offspring chromosome
    int dim, float alpha                            // Chromosome dimension, crossover weight
) {{
    for (int i = 0; i < dim; ++i) {{
        // Produces one offspring per cell typically in cGA
        offspring_chromo[i] = alpha * p1_chromo[i] + (1.0f - alpha) * p2_chromo[i];
    }}
}}

// Example: Gaussian Mutation for continuous chromosomes
__device__ void gaussian_mutation_gpu(
    float* chromo, int dim,                         // Chromosome to mutate, its dimension
    float mutation_rate_gene, float mutation_strength, // Mutation parameters
    float lower_bound, float upper_bound,           // Gene value bounds
    curandState* rand_state                         // cuRAND state for this thread
) {{
    for (int i = 0; i < dim; ++i) {{
        if (curand_uniform(rand_state) < mutation_rate_gene) {{ // Per-gene mutation check
            // Generate Gaussian random number N(0,1), then scale by mutation_strength
            float mutation_val = curand_normal(rand_state) * mutation_strength;
            chromo[i] += mutation_val;
            // Clamp to bounds
            if (chromo[i] < lower_bound) chromo[i] = lower_bound;
            if (chromo[i] > upper_bound) chromo[i] = upper_bound;
        }}
    }}
}}
// --- END Helper __device__ functions ---


__global__ void cga_interaction_kernel(
    const float* current_chromosomes_flat, // Gen t: Chromosomes
    const float* current_fitness_values,   // Gen t: Fitness values
    float* new_chromosomes_flat,           // Output: Gen t+1 Chromosomes
    int grid_width, int grid_height, int chromosome_dim,
    float crossover_rate_individual, float mutation_rate_gene, float mutation_strength,
    float lower_bound, float upper_bound,
    int neighborhood_type, // 0 for Moore (8 neighbors), 1 for Von Neumann (4 neighbors)
    curandState* rand_states_global // Array of cuRAND states, one per thread
) {{
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x; // Global unique ID for each cell/individual
    
    if (flat_idx >= grid_width * grid_height) return; // Out of bounds check

    // Get cuRAND state for the current thread
    curandState local_rand_state = rand_states_global[flat_idx];

    // Current individual (parent 1)
    const float* p1_chromo = current_chromosomes_flat + flat_idx * chromosome_dim;
    
    // --- 1. Neighborhood Definition & Parent (Mate) Selection ---
    // The current cell's individual (p1_chromo) is one parent.
    // Select a mate from its neighborhood.
    // Example: Randomly select one neighbor. More sophisticated: tournament among neighbors.
    
    int r = flat_idx / grid_width; // Current cell's row
    int c = flat_idx % grid_width; // Current cell's column

    // Moore neighborhood offsets (excluding self)
    int dr_moore[] = {{-1, -1, -1,  0, 0,  1, 1, 1}};
    int dc_moore[] = {{-1,  0,  1, -1, 1, -1, 0, 1}};
    // Von Neumann neighborhood offsets (excluding self)
    int dr_vn[] = {{-1, 1, 0, 0}};
    int dc_vn[] = {{0, 0, -1, 1}};

    int* p_dr = (neighborhood_type == 0) ? dr_moore : dr_vn;
    int* p_dc = (neighborhood_type == 0) ? dc_moore : dc_vn;
    int num_potential_neighbors = (neighborhood_type == 0) ? 8 : 4;

    // Collect valid neighbor indices (toroidal grid assumed)
    int valid_neighbor_indices[8]; // Max 8 neighbors for Moore
    int valid_neighbor_count = 0;
    for (int i = 0; i < num_potential_neighbors; ++i) {{
        int nr = (r + p_dr[i] + grid_height) % grid_height; // Toroidal wrap-around
        int nc = (c + p_dc[i] + grid_width) % grid_width;   // Toroidal wrap-around
        valid_neighbor_indices[valid_neighbor_count++] = nr * grid_width + nc;
    }}
    
    int mate_flat_idx = flat_idx; // Default to self (clone if no valid neighbors or no crossover)
    if (valid_neighbor_count > 0) {{
        // Select a random neighbor as mate
        int random_choice = (int)(curand_uniform(&local_rand_state) * valid_neighbor_count);
        mate_flat_idx = valid_neighbor_indices[random_choice % valid_neighbor_count]; // Modulo for safety
    }}
    const float* p2_chromo = current_chromosomes_flat + mate_flat_idx * chromosome_dim;
    
    // Offspring chromosome buffer (points directly into new_chromosomes_flat)
    float* offspring_chromo = new_chromosomes_flat + flat_idx * chromosome_dim;

    // --- 2. Crossover ---
    if (curand_uniform(&local_rand_state) < crossover_rate_individual) {{
        float alpha = curand_uniform(&local_rand_state); // Alpha for arithmetic crossover
        arithmetic_crossover_gpu(p1_chromo, p2_chromo, offspring_chromo, chromosome_dim, alpha);
    }} else {{
        // No crossover, clone parent1 (current cell's individual) to offspring
        for (int i = 0; i < chromosome_dim; ++i) {{
            offspring_chromo[i] = p1_chromo[i];
        }}
    }}

    // --- 3. Mutation ---
    // Mutate the offspring chromosome that is now in `offspring_chromo`
    gaussian_mutation_gpu(offspring_chromo, chromosome_dim, mutation_rate_gene, mutation_strength,
                          lower_bound, upper_bound, &local_rand_state);
    
    // Save updated cuRAND state for the next generation/kernel launch
    rand_states_global[flat_idx] = local_rand_state;
}}
"""


class CellularGA(BaseGA):
    def __init__(self, problem: BenchmarkProblem,
                 grid_width: int, grid_height: int, 
                 generations: int,
                 crossover_rate: float, mutation_rate_gene: float, 
                 neighborhood_type: str = "moore", 
                 mutation_strength: float = 0.1,
                 verbose: bool = True):

        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA is not available or CUDA context failed. CellularGA cannot be used.")
        if problem.problem_type != "continuous":
            logging.warning("Current CellularGA CUDA kernels are primarily designed for continuous problems. Functionality for other types may be limited.")
            # raise NotImplementedError("CellularGA with PyCUDA currently optimized for continuous problems.")


        population_size = grid_width * grid_height
        
        # BaseGA expects operator objects. For CellularGA, these are "embedded" in CUDA kernels.
        # We pass dummy operators to BaseGA constructor.
        class DummyOp:
            def apply(self, *args, **kwargs): pass
        super().__init__(problem, population_size, generations, 
                         DummyOp(), DummyOp(), DummyOp(), 
                         elite_size=0, # Elitism is not typical in basic cGAs this way
                         verbose=verbose)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.crossover_rate_gpu = np.float32(crossover_rate) # Per-individual rate for interaction kernel
        self.mutation_rate_gene_gpu = np.float32(mutation_rate_gene) # Per-gene rate
        self.mutation_strength_gpu = np.float32(mutation_strength)
        self.lower_bound_gpu = np.float32(problem.value_bounds[0])
        self.upper_bound_gpu = np.float32(problem.value_bounds[1])
        
        self.neighborhood_type_map = {"moore": 0, "von_neumann": 1}
        if neighborhood_type.lower() not in self.neighborhood_type_map:
            raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")
        self.neighborhood_type_gpu_val = np.int32(self.neighborhood_type_map[neighborhood_type.lower()])

        self.chromosome_dim = self.problem.get_chromosome_length()

        # GPU memory buffers
        self.d_chromosomes = None      # Device memory for current generation chromosomes
        self.d_fitness_values = None # Device memory for fitness values
        self.d_new_chromosomes = None  # Device memory for next generation chromosomes
        self.d_rand_states = None      # Device memory for cuRAND states

        self._compile_kernels()
        self._initialize_gpu_buffers_and_rng()
        
        if self.verbose:
            logging.info(f"Cellular GA initialized for a {grid_width}x{grid_height} grid. Chromosome dim: {self.chromosome_dim}.")

    def _compile_kernels(self):
        # TODO: Implement robust templating or dynamic kernel generation for different problems.
        # For now, using the predefined kernel strings.
        try:
            # Concatenate all kernel code that might have __device__ functions used by __global__ kernels
            full_kernel_code = CUDA_FITNESS_KERNEL_TEMPLATE + "\n" + CUDA_INTERACTION_KERNEL_TEMPLATE
            cuda_inc = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"

            self.module = SourceModule(
                full_kernel_code,
                include_dirs=[cuda_inc],
                # wrap it in quotes so the shell sees it as one argument
                options=[f'-I"{cuda_inc}"']
            )

            self.fitness_kernel_gpu = self.module.get_function("evaluate_fitness_kernel")
            self.interaction_kernel_gpu = self.module.get_function("cga_interaction_kernel")
        except Exception as e:
            logging.error(f"Error compiling CUDA kernels: {e}", exc_info=True)
            logging.error("Ensure CUDA toolkit is installed and PyCUDA can find nvcc.")
            logging.error("The kernel code itself might also have syntax issues.")
            raise

    def _initialize_gpu_buffers_and_rng(self):
        # Initialize population on CPU first to get chromosomes
        super()._initialize_population() # Creates self.population with Individual objects

        h_initial_chromosomes = np.array([ind.chromosome for ind in self.population.individuals], dtype=np.float32).flatten()
        h_initial_fitness = np.full(self.population_size, float('inf'), dtype=np.float32)

        # Allocate GPU memory
        self.d_chromosomes = cuda.mem_alloc(h_initial_chromosomes.nbytes)
        cuda.memcpy_htod(self.d_chromosomes, h_initial_chromosomes)
        
        self.d_fitness_values = cuda.mem_alloc(h_initial_fitness.nbytes)
        # cuda.memcpy_htod(self.d_fitness_values, h_initial_fitness) # Fitness computed on GPU

        self.d_new_chromosomes = cuda.mem_alloc(h_initial_chromosomes.nbytes) # Buffer for offspring

        # Initialize cuRAND states on GPU
        # CRITICAL: Proper RNG setup is vital. PyCUDA's curandom.XORWOWRandomNumberGenerator
        # provides a high-level way to manage states and generate random numbers.
        # The kernel template uses raw curandState* which requires manual setup.
        # For simplicity and robustness, using PyCUDA's high-level generator is preferred if not writing raw states kernel.
        # Let's use the high-level generator approach here instead of manual state kernel.
        # This means the interaction kernel would need to be adapted to use generator.fill_uniform(), etc.
        # OR, we stick to the manual curandState and write the init kernel.
        # Given the prompt's direction towards CUDA kernels, I'll assume manual state management is intended.
        
        # --- Manual cuRAND State Initialization ---
        # This requires a kernel to call curand_init() for each state.
        # Create and compile the RNG initialization kernel
        rng_init_kernel_code = """
        #include <curand_kernel.h>
        extern "C" { // Needed if function name might be mangled by C++ compiler for PyCUDA
        __global__ void initialize_rng_states(unsigned long long seed, int num_states, curandState* states) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_states) {
                // Initialize each state with a unique sequence: seed + idx ensures different streams
                curand_init(seed + idx, 0, 0, &states[idx]);
            }
        }}
        """
        try:
            rng_module = SourceModule(rng_init_kernel_code)
            init_rng_func = rng_module.get_function("initialize_rng_states")

            # Allocate memory for cuRAND states on GPU
            # Size of curandStateXORWOW (default for curand_init with 3 args)
            # This size can vary. PyCUDA's curandom might have utils for this, or check CUDA docs.
            # Assuming direct N*sizeof(curandState) is okay if kernel defines curandState.
            # For safety, let's estimate. Typically around 24-48 bytes. Let's use a safe size.
            # PyCUDA's curandom.py shows `curandStateXORWOW_dtype` which has itemsize.
            # If using raw `curandState*` in kernel, you must know its size or use opaque pointers.
            # Let's use pycuda.curandom to get the state size.
            rng_state_size = curandom.XORWOWRandomNumberGenerator().state_dtype.itemsize # Gets size of one state
            self.d_rand_states = cuda.mem_alloc(self.population_size * rng_state_size)
            
            threads_per_block_rng = 256
            blocks_per_grid_rng = (self.population_size + threads_per_block_rng - 1) // threads_per_block_rng
            
            current_seed = np.uint64(int(time.time())) # Seed for RNG
            init_rng_func(current_seed, np.int32(self.population_size), self.d_rand_states,
                          block=(threads_per_block_rng,1,1), grid=(blocks_per_grid_rng,1))
            cuda.Context.synchronize() # Ensure RNG init is complete
            if self.verbose: logging.info(f"cuRAND states initialized on GPU with seed {current_seed}.")
        except Exception as e:
            logging.error(f"Failed to initialize cuRAND states on GPU: {e}", exc_info=True)
            logging.error("This is a critical step for CellularGA. Check CUDA setup and kernel code.")
            # Potentially free allocated memory if error occurs early
            if self.d_rand_states: self.d_rand_states.free()
            self.d_rand_states = None
            raise RuntimeError("cuRAND state initialization failed.") from e


    def _evaluate_population(self, population_ignored: Population = None):
        # This method is called by BaseGA's run loop.
        # For CellularGA, it runs the fitness evaluation kernel on GPU.
        # `population_ignored` is not used as data resides on GPU (self.d_chromosomes).
        if not self.d_chromosomes or not self.d_fitness_values or not self.fitness_kernel_gpu:
            logging.error("CellularGA: GPU buffers or kernel not initialized for evaluation.")
            return

        threads_per_block = 256 
        blocks_per_grid = (self.population_size + threads_per_block - 1) // threads_per_block

        self.fitness_kernel_gpu(
            self.d_chromosomes, self.d_fitness_values,
            np.int32(self.population_size), np.int32(self.chromosome_dim),
            block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1)
        )
        # cuda.Context.synchronize() # Synchronization usually happens before reading data back or next kernel if dependent.

    def _run_cga_evolution_step_on_gpu(self):
        """ Performs one generation of cGA evolution (selection, crossover, mutation) on GPU."""
        if not all([self.d_chromosomes, self.d_fitness_values, self.d_new_chromosomes, self.d_rand_states, self.interaction_kernel_gpu]):
            logging.error("CellularGA: GPU buffers or interaction kernel not ready for evolution step.")
            return

        threads_per_block = 256
        blocks_per_grid = (self.population_size + threads_per_block - 1) // threads_per_block

        self.interaction_kernel_gpu(
            self.d_chromosomes, self.d_fitness_values, self.d_new_chromosomes,
            np.int32(self.grid_width), np.int32(self.grid_height), np.int32(self.chromosome_dim),
            self.crossover_rate_gpu, self.mutation_rate_gene_gpu, self.mutation_strength_gpu,
            self.lower_bound_gpu, self.upper_bound_gpu,
            self.neighborhood_type_gpu_val,
            self.d_rand_states, # Pass initialized RNG states
            block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1)
        )
        # cuda.Context.synchronize() # Ensure interaction kernel is complete before swapping buffers

        # Swap chromosome buffers: new generation becomes current generation for next iteration
        self.d_chromosomes, self.d_new_chromosomes = self.d_new_chromosomes, self.d_chromosomes

    def run(self):
        if not PYCUDA_AVAILABLE or not self.d_rand_states : # Check if RNG init was successful too
            logging.error("CellularGA cannot run due to PyCUDA issues or RNG initialization failure.")
            return None, [], []

        # Initial population is already on GPU from _initialize_gpu_buffers_and_rng()
        # Evaluate the initial population on GPU
        self._evaluate_population() 

        for gen in range(self.generations):
            self.current_generation = gen
            cuda.Context.synchronize() # Ensure previous kernel (eval or interaction) is done before logging or next step

            # --- Logging statistics (requires data transfer GPU -> CPU, can be slow) ---
            if self.verbose and (gen % 10 == 0 or gen == self.generations - 1 or self.problem.is_optimal_solution_known_for_logging()):
                h_fitness_values_for_log = np.empty(self.population_size, dtype=np.float32)
                cuda.memcpy_dtoh(h_fitness_values_for_log, self.d_fitness_values)
                
                current_best_fitness_gpu = np.min(h_fitness_values_for_log)
                current_avg_fitness_gpu = np.mean(h_fitness_values_for_log)
                
                self.best_fitness_over_time.append(float(current_best_fitness_gpu)) # Cast to standard float
                self.avg_fitness_over_time.append(float(current_avg_fitness_gpu))

                logging.info(f"Generation {gen}: Best Fitness (GPU) = {current_best_fitness_gpu:.4f}, Avg Fitness (GPU) = {current_avg_fitness_gpu:.4f}")

                if self.problem.is_optimal_solution(current_best_fitness_gpu):
                    logging.info(f"Optimal solution potentially found at generation {gen} on GPU.")
                    break 
            
            # --- cGA Evolution Step on GPU (Selection, Crossover, Mutation) ---
            self._run_cga_evolution_step_on_gpu()
            
            # --- Evaluate the new population (now in self.d_chromosomes after swap) ---
            self._evaluate_population() # Calculates fitness for the new generation on GPU

        cuda.Context.synchronize() # Ensure all GPU work is done before final data retrieval
        
        # Retrieve the best individual from GPU after all generations
        h_final_chromosomes_flat = np.empty(self.population_size * self.chromosome_dim, dtype=np.float32)
        h_final_fitness = np.empty(self.population_size, dtype=np.float32)
        cuda.memcpy_dtoh(h_final_chromosomes_flat, self.d_chromosomes)
        cuda.memcpy_dtoh(h_final_fitness, self.d_fitness_values)

        h_final_chromosomes_reshaped = h_final_chromosomes_flat.reshape(self.population_size, self.chromosome_dim)

        best_idx_on_gpu = np.argmin(h_final_fitness)
        best_chromosome_on_gpu = h_final_chromosomes_reshaped[best_idx_on_gpu, :].tolist() # Convert to list
        best_fitness_on_gpu = float(h_final_fitness[best_idx_on_gpu])

        # Create an Individual object for the result
        final_best_individual = Individual(best_chromosome_on_gpu, problem_type=self.problem.problem_type)
        final_best_individual.fitness = best_fitness_on_gpu
        
        if self.verbose:
            logging.info(f"Finished Cellular GA run. Best individual from GPU: {final_best_individual}")

        # Free GPU memory explicitly (though autoinit helps on exit, good practice for long runs)
        try:
            if self.d_chromosomes: self.d_chromosomes.free()
            if self.d_fitness_values: self.d_fitness_values.free()
            if self.d_new_chromosomes: self.d_new_chromosomes.free()
            if self.d_rand_states: self.d_rand_states.free()
        except cuda.LogicError as e: # Handles cases where context might already be torn down
            logging.warning(f"Minor error during GPU memory cleanup: {e}")


        return final_best_individual, self.best_fitness_over_time, self.avg_fitness_over_time