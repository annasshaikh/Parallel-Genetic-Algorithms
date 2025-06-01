# ga/plotting.py
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting (saves to file)
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging

# Get a logger for this module
logger = logging.getLogger("Plotting")

def plot_convergence(fitness_log_file: str, output_dir: str, experiment_name: str):
    """
    Plots convergence (best and average fitness vs. generation) from a CSV log file.
    Saves the plot to the specified output directory.
    """
    try:
        if not os.path.exists(fitness_log_file):
            logger.warning(f"Fitness log file not found: {fitness_log_file}. Skipping convergence plot.")
            return

        df = pd.read_csv(fitness_log_file)
        if df.empty or "Generation" not in df.columns or "BestFitness" not in df.columns:
            logger.warning(f"Fitness log file {fitness_log_file} is empty or has incorrect format. Skipping convergence plot.")
            return

        plt.figure(figsize=(12, 7)) # Slightly larger figure
        
        # Plot Best Fitness
        if 'BestFitness' in df.columns and not df['BestFitness'].isnull().all():
            plt.plot(df["Generation"], df["BestFitness"], label="Best Fitness", color="dodgerblue", linewidth=2)
        
        # Plot Average Fitness if available
        if "AverageFitness" in df.columns and not df['AverageFitness'].isnull().all():
            plt.plot(df["Generation"], df["AverageFitness"], label="Average Fitness", color="darkorange", linestyle="--", linewidth=1.5)
        
        plt.title(f"Convergence Plot: {experiment_name}", fontsize=16)
        plt.xlabel("Generation", fontsize=14)
        plt.ylabel("Fitness", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout() # Adjust plot to prevent labels from being clipped
        
        plot_filename = os.path.join(output_dir, f"convergence_plot_{experiment_name}.png")
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        logger.info(f"Convergence plot saved to {plot_filename}")

    except Exception as e:
        logger.error(f"Error generating convergence plot for {experiment_name} from {fitness_log_file}: {e}", exc_info=True)


def plot_speedup_efficiency(results_dir: str, experiment_group_name: str, 
                            baseline_problem_runtime_map: dict, 
                            parallel_runs_data: list):
    """
    Generates speedup and parallel efficiency plots and saves data to CSV.
    This function assumes a specific structure for comparison.

    Args:
        results_dir (str): Base directory to save plots and CSV.
        experiment_group_name (str): Name for this group of comparisons (e.g., 'Sphere_Problem_Speedup').
        baseline_problem_runtime_map (dict): Maps problem_name to its serial baseline runtime.
                                             Example: {"Sphere_dim10": 120.5, "TSP_20": 60.2}
        parallel_runs_data (list): A list of dictionaries, each representing a parallel run's metrics.
                                   Required keys: "problem_name", "num_workers_or_grid_size", "runtime_seconds".
                                   Example: [{"problem_name": "Sphere_dim10", "num_workers_or_grid_size": 4, "runtime_seconds": 35.2}, ...]
    """
    if not parallel_runs_data or not baseline_problem_runtime_map:
        logger.warning(f"Insufficient data for speedup/efficiency plots for group '{experiment_group_name}'. Baseline or parallel runtimes missing/invalid.")
        return

    # Group parallel runs by problem_name
    problem_grouped_runs = {}
    for run_data in parallel_runs_data:
        prob_name = run_data.get("problem_name") # Ensure your metrics include this
        if prob_name:
            if prob_name not in problem_grouped_runs:
                problem_grouped_runs[prob_name] = []
            problem_grouped_runs[prob_name].append(run_data)

    for problem_name, runs in problem_grouped_runs.items():
        baseline_runtime = baseline_problem_runtime_map.get(problem_name)
        if baseline_runtime is None or baseline_runtime <= 0:
            logger.warning(f"No valid baseline runtime found for problem '{problem_name}' in group '{experiment_group_name}'. Skipping speedup plots for this problem.")
            continue

        # Prepare data for this specific problem
        plot_data = []
        for run in sorted(runs, key=lambda x: x["num_workers_or_grid_size"]): # Sort by num_workers
            processors = run["num_workers_or_grid_size"]
            parallel_runtime = run["runtime_seconds"]
            
            if not isinstance(processors, (int, float)) or processors <= 0 or parallel_runtime <=0: # Ensure processors is a number
                logger.debug(f"Skipping invalid run data for problem '{problem_name}': processors={processors}, runtime={parallel_runtime}")
                continue


            speedup = baseline_runtime / parallel_runtime
            efficiency = speedup / processors
            plot_data.append({
                "Processors": processors,
                "Runtime": parallel_runtime,
                "Speedup": speedup,
                "Efficiency": efficiency
            })
        
        if not plot_data:
            logger.info(f"No valid parallel run data to plot for problem '{problem_name}' in group '{experiment_group_name}'.")
            continue

        df = pd.DataFrame(plot_data)
        
        # Ensure results directory for this specific group/problem exists
        output_subdir = os.path.join(results_dir, "comparative_plots", f"{experiment_group_name}_{problem_name.replace('/','_')}") # Sanitize problem_name
        os.makedirs(output_subdir, exist_ok=True)

        csv_filename = os.path.join(output_subdir, f"speedup_efficiency_data.csv")
        df.to_csv(csv_filename, index=False)
        logger.info(f"Speedup/Efficiency data for '{problem_name}' saved to {csv_filename}")

        # Plot Speedup
        plt.figure(figsize=(10, 6))
        plt.plot(df["Processors"], df["Speedup"], marker='o', linestyle='-', color='blue', label="Actual Speedup")
        plt.plot(df["Processors"], df["Processors"], linestyle='--', color='gray', label="Ideal Speedup")
        plt.title(f"Speedup Curve: {experiment_group_name} - {problem_name}", fontsize=15)
        plt.xlabel("Number of Processors/Parallel Units", fontsize=12)
        plt.ylabel("Speedup (T_serial / T_parallel)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        unique_processors = sorted(df["Processors"].unique())
        if len(unique_processors) > 1 : 
            if all(isinstance(p, (int, float)) for p in unique_processors): # Check if numeric for ticks
                plt.xticks(unique_processors)
            else: # If processor identifiers are strings (like "16x16"), handle differently or skip specific ticks
                logger.debug("Processor identifiers are not all numeric; using default x-axis ticks for speedup plot.")


        speedup_plot_filename = os.path.join(output_subdir, f"speedup_plot.png")
        plt.savefig(speedup_plot_filename)
        plt.close()
        logger.info(f"Speedup plot for '{problem_name}' saved to {speedup_plot_filename}")

        # Plot Efficiency
        plt.figure(figsize=(10, 6))
        plt.plot(df["Processors"], df["Efficiency"], marker='s', linestyle='-', color='green', label="Parallel Efficiency")
        plt.axhline(1.0, linestyle='--', color='gray', label="Ideal Efficiency (100%)") # Ideal efficiency line
        plt.title(f"Parallel Efficiency: {experiment_group_name} - {problem_name}", fontsize=15)
        plt.xlabel("Number of Processors/Parallel Units", fontsize=12)
        plt.ylabel("Efficiency (Speedup / Processors)", fontsize=12)
        plt.ylim(0, max(1.2, df["Efficiency"].max() * 1.1 if not df["Efficiency"].empty else 1.2)) # Dynamic upper Y limit
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)

        if len(unique_processors) > 1:
            if all(isinstance(p, (int, float)) for p in unique_processors):
                plt.xticks(unique_processors)
            else:
                logger.debug("Processor identifiers are not all numeric; using default x-axis ticks for efficiency plot.")
        
        efficiency_plot_filename = os.path.join(output_subdir, f"efficiency_plot.png")
        plt.savefig(efficiency_plot_filename)
        plt.close()
        logger.info(f"Efficiency plot for '{problem_name}' saved to {efficiency_plot_filename}")