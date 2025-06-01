# ga/experiments.py
import logging
import time
import os
from typing import Dict, List, Any, Callable

from .base import (Individual, Population, BaseGA,
                   TournamentSelection, ArithmeticCrossover, TwoPointCrossover, OrderCrossoverTSP,
                   GaussianMutation, BitFlipMutation, SwapMutationTSP)
from .benchmarks import (BenchmarkProblem, Sphere, Rastrigin, Griewank,
                         TravelingSalesmanProblem, RoyalRoad)
from .master_slave import MasterSlaveGA
from .cellular_cuda import CellularGA, PYCUDA_AVAILABLE
from .utils import Timer, save_results, get_cpu_gpu_utilization
from .plotting import plot_convergence


# Get a logger for this module
logger = logging.getLogger("Experiments")

# --- Helper functions to instantiate problems and operators from config strings ---
def get_problem_instance(problem_config: Dict[str, Any]) -> BenchmarkProblem:
    name = problem_config.get("name", "").lower()
    params = problem_config.get("params", {})
    target_fitness = params.get("target_fitness", None)

    if name == "sphere":
        return Sphere(dimensions=params.get("dimensions", 10), 
                      value_bounds=tuple(params.get("value_bounds", [-5.12, 5.12])),
                      target_fitness=target_fitness)
    elif name == "rastrigin":
        return Rastrigin(dimensions=params.get("dimensions", 10),
                         value_bounds=tuple(params.get("value_bounds", [-5.12, 5.12])),
                         target_fitness=target_fitness)
    elif name == "griewank":
        return Griewank(dimensions=params.get("dimensions", 10),
                        value_bounds=tuple(params.get("value_bounds", [-600.0, 600.0])),
                        target_fitness=target_fitness)
    elif name == "tsp":
        # For TSP, distance_matrix could be loaded from a file or generated
        # For simplicity here, it generates a random one if not provided.
        # In a real setup, you might pass a path to a TSP file in params.
        return TravelingSalesmanProblem(num_cities=params.get("num_cities", 20),
                                        distance_matrix=params.get("distance_matrix", None), # Or load from file
                                        target_fitness=target_fitness)
    elif name == "royalroad":
        return RoyalRoad(block_size=params.get("block_size", 8),
                         num_blocks=params.get("num_blocks", 4),
                         target_fitness_minimization=target_fitness) # Target for min version
    else:
        raise ValueError(f"Unknown problem name: {name}")

def get_operators(operator_configs: Dict[str, Dict[str, Any]], problem: BenchmarkProblem) -> Dict[str, Any]:
    operators = {}
    
    # Selection
    sel_config = operator_configs.get("selection", {})
    sel_type = sel_config.get("type", "tournament").lower()
    if sel_type == "tournament":
        operators["selection_op"] = TournamentSelection(tournament_size=sel_config.get("tournament_size", 3))
    else:
        raise ValueError(f"Unsupported selection type: {sel_type}")

    # Crossover
    cross_config = operator_configs.get("crossover", {})
    cross_type = cross_config.get("type", "").lower() # Default depends on problem
    cross_rate = cross_config.get("rate", 0.8)
    
    if not cross_type: # Auto-detect based on problem
        if problem.problem_type == "continuous":
            cross_type = "arithmetic"
        elif problem.problem_type == "binary":
            cross_type = "two_point"
        elif problem.problem_type == "combinatorial_permutation":
            cross_type = "order_tsp"
        else:
            raise ValueError(f"Cannot auto-detect crossover for problem type: {problem.problem_type}")
            
    if cross_type == "arithmetic":
        operators["crossover_op"] = ArithmeticCrossover(crossover_rate=cross_rate)
    elif cross_type == "two_point":
        operators["crossover_op"] = TwoPointCrossover(crossover_rate=cross_rate)
    elif cross_type == "order_tsp":
        operators["crossover_op"] = OrderCrossoverTSP(crossover_rate=cross_rate)
    else:
        raise ValueError(f"Unsupported crossover type: {cross_type}")

    # Mutation
    mut_config = operator_configs.get("mutation", {})
    mut_type = mut_config.get("type", "").lower() # Default depends on problem
    mut_rate = mut_config.get("rate", 0.01) # Often per-gene for continuous/binary, per-individual for TSP
    
    if not mut_type: # Auto-detect
        if problem.problem_type == "continuous":
            mut_type = "gaussian"
        elif problem.problem_type == "binary":
            mut_type = "bit_flip"
        elif problem.problem_type == "combinatorial_permutation":
            mut_type = "swap_tsp"
        else:
            raise ValueError(f"Cannot auto-detect mutation for problem type: {problem.problem_type}")

    if mut_type == "gaussian":
        operators["mutation_op"] = GaussianMutation(mutation_rate=mut_rate, # Per-gene
                                                    mutation_strength=mut_config.get("strength", 0.1),
                                                    value_bounds=problem.value_bounds)
    elif mut_type == "bit_flip":
        operators["mutation_op"] = BitFlipMutation(mutation_rate=mut_rate) # Per-gene
    elif mut_type == "swap_tsp":
        operators["mutation_op"] = SwapMutationTSP(mutation_rate=mut_config.get("rate", 0.1)) # Per-individual
    else:
        raise ValueError(f"Unsupported mutation type: {mut_type}")
        
    return operators


class ExperimentRunner:
    def __init__(self, config: Dict[str, Any], results_base_dir: str = "results"):
        self.config = config
        self.results_base_dir = results_base_dir
        os.makedirs(self.results_base_dir, exist_ok=True)

        self.global_settings = config.get("global_settings", {})
        self.verbose = self.global_settings.get("verbose", True)

    def run_single_experiment(self, experiment_config: Dict[str, Any], trial_num: int) -> Dict[str, Any]:
        exp_name = experiment_config.get("name", f"Experiment_Trial{trial_num}")
        model_type = experiment_config.get("model", "serial_ga").lower()
        
        problem_conf = experiment_config["problem"]
        problem_instance = get_problem_instance(problem_conf)
        problem_instance.reset_evaluations() # Ensure fresh count for each run

        ga_params = experiment_config.get("ga_parameters", {})
        generations = ga_params.get("generations", 100)
        population_size = ga_params.get("population_size", 50) # For MS and Serial
        elite_size = ga_params.get("elite_size", 1)

        # Specific model parameters
        model_params = experiment_config.get("model_parameters", {})

        logger.info(f"--- Starting Experiment: {exp_name} (Trial {trial_num}) ---")
        logger.info(f"Model: {model_type}, Problem: {problem_instance.name} (Dim/Size: {problem_instance.get_chromosome_length()})")
        logger.info(f"Generations: {generations}, Population Size: {population_size if model_type != 'cellular_cuda' else model_params.get('grid_width',10)*model_params.get('grid_height',10)}")

        ga_model: BaseGA = None
        timer = Timer(logger=logger)
        
        with timer:
            if model_type == "serial_ga" or model_type == "master_slave":
                operator_details = get_operators(experiment_config.get("operators", {}), problem_instance)
                common_args = {
                    "problem": problem_instance,
                    "population_size": population_size,
                    "generations": generations,
                    "selection_op": operator_details["selection_op"],
                    "crossover_op": operator_details["crossover_op"],
                    "mutation_op": operator_details["mutation_op"],
                    "elite_size": elite_size,
                    "verbose": self.verbose
                }
                if model_type == "master_slave":
                    ga_model = MasterSlaveGA(**common_args, num_workers=model_params.get("num_workers"))
                else: # serial_ga
                    # To make it truly serial for baseline, MasterSlaveGA with num_workers=1
                    ga_model = MasterSlaveGA(**common_args, num_workers=1) 
            
            elif model_type == "cellular_cuda":
                if not PYCUDA_AVAILABLE:
                    logger.error("CellularGA (CUDA) selected, but PyCUDA is not available. Skipping this experiment.")
                    return {
                        "name": exp_name, "trial": trial_num, "status": "skipped", 
                        "reason": "PyCUDA not available", "problem_name": problem_instance.name, 
                         "num_workers_or_grid_size": f"{model_params.get('grid_width')}x{model_params.get('grid_height')}"
                    }
                
                # CellularGA takes rates directly, not operator objects
                op_configs_for_cga = experiment_config.get("operators", {})
                crossover_conf_cga = op_configs_for_cga.get("crossover", {})
                mutation_conf_cga = op_configs_for_cga.get("mutation", {})

                ga_model = CellularGA(
                    problem=problem_instance,
                    grid_width=model_params.get("grid_width", 10),
                    grid_height=model_params.get("grid_height", 10),
                    generations=generations,
                    crossover_rate=crossover_conf_cga.get("rate", 0.8), # This is individual rate for cGA
                    mutation_rate_gene=mutation_conf_cga.get("rate", 0.01), # This is gene-level rate for cGA
                    mutation_strength=mutation_conf_cga.get("mutation",{}).get("strength", 0.1), # If Gaussian
                    neighborhood_type=model_params.get("neighborhood", "moore"),
                    verbose=self.verbose
                )
            else:
                raise ValueError(f"Unsupported GA model type: {model_type}")

            best_individual, best_fitness_log, avg_fitness_log = ga_model.run()
        
        run_time = timer.elapsed_time
        utilization = get_cpu_gpu_utilization(logger=logger) # Get utilization after the run

        metrics = {
            "name": exp_name,
            "trial": trial_num,
            "model_type": model_type,
            "problem_name": problem_instance.name,
            "problem_dimensions": problem_instance.get_chromosome_length(),
            "population_size_config": population_size if model_type != 'cellular_cuda' else f"{model_params.get('grid_width')}x{model_params.get('grid_height')}",
            "num_workers_or_grid_size": model_params.get("num_workers", 1) if model_type == "master_slave" else (f"{model_params.get('grid_width')}x{model_params.get('grid_height')}" if model_type == "cellular_cuda" else 1),
            "generations_run": ga_model.current_generation + 1, # current_generation is 0-indexed
            "target_generations": generations,
            "best_fitness_achieved": best_individual.fitness if best_individual else float('inf'),
            "total_fitness_evaluations": problem_instance.evaluations,
            "runtime_seconds": run_time,
            "cpu_utilization_percent": utilization.get("cpu_percent", "N/A"),
            "gpu_utilization_percent": utilization.get("gpu_percent_util", "N/A"),
            "status": "completed"
        }
        if best_individual and problem_instance.target_fitness is not None:
            metrics["reached_target_fitness"] = problem_instance.is_optimal_solution(best_individual.fitness)
        
        # Save detailed results for this specific run
        experiment_run_path = save_results(
            results_dir=self.results_base_dir,
            experiment_name=f"{exp_name}_Trial{trial_num}_{model_type}",
            metrics=metrics,
            config=experiment_config, # Save the config portion for this run
            best_fitness_log=best_fitness_log,
            avg_fitness_log=avg_fitness_log,
            logger=logger
        )
        
        if experiment_run_path and best_fitness_log: # Plot if path and logs exist
            plot_convergence(
                fitness_log_file=os.path.join(experiment_run_path, "fitness_log.csv"),
                output_dir=experiment_run_path,
                experiment_name=f"{exp_name}_Trial{trial_num}_{model_type}"
            )
        
        logger.info(f"--- Finished Experiment: {exp_name} (Trial {trial_num}) --- Runtime: {run_time:.2f}s, Best Fitness: {metrics['best_fitness_achieved']:.4f}\n")
        return metrics


    def run_all_experiments(self) -> List[Dict[str, Any]]:
        all_experiment_metrics = []
        if "experiments" not in self.config or not self.config["experiments"]:
            logger.warning("No experiments defined in the configuration file.")
            return all_experiment_metrics

        for i, exp_conf_item in enumerate(self.config["experiments"]):
            num_trials = exp_conf_item.get("trials", 1)
            experiment_base_name = exp_conf_item.get("name", f"UnnamedExperiment_{i}")

            for trial in range(1, num_trials + 1):
                try:
                    # Deep copy or careful handling if configs are modified per trial later
                    current_exp_config = exp_conf_item.copy() # Shallow copy is usually fine if not modifying nested dicts
                    current_exp_config["name"] = f"{experiment_base_name}" # Keep base name, trial handled in run_single_experiment
                    
                    metrics = self.run_single_experiment(current_exp_config, trial)
                    all_experiment_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error running trial {trial} for experiment '{experiment_base_name}': {e}", exc_info=True)
                    all_experiment_metrics.append({
                        "name": experiment_base_name, "trial": trial, "status": "error", 
                        "reason": str(e), "problem_name": exp_conf_item.get("problem",{}).get("name", "Unknown"),
                        "num_workers_or_grid_size": exp_conf_item.get("model_parameters",{}).get("num_workers", "N/A") if exp_conf_item.get("model") == "master_slave" else (f"{exp_conf_item.get('model_parameters',{}).get('grid_width','?')}x{exp_conf_item.get('model_parameters',{}).get('grid_height','?')}" if exp_conf_item.get("model") == "cellular_cuda" else "N/A")
                    })
        
        # After all experiments, one could generate comparative plots (speedup/efficiency)
        # This requires collecting baseline (serial) runtimes and parallel runtimes.
        # For now, this part is left for manual analysis or a separate script using the saved CSVs/JSONs.
        # `plot_speedup_efficiency` from plotting.py can be used here if data is aggregated correctly.

        return all_experiment_metrics