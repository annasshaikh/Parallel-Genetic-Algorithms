# main.py
import argparse
import logging
import os
import json
from datetime import datetime

from ga.utils import load_config, setup_logging
from ga.experiments import ExperimentRunner
from ga.plotting import plot_speedup_efficiency # For potential comparative plots

def main():
    parser = argparse.ArgumentParser(description="Parallel Genetic Algorithms Framework")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML/JSON configuration file for experiments."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory to save experiment results (logs, plots, metrics)."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save main log files."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )
    
    args = parser.parse_args()

    # Setup main logging (distinct from per-experiment logs if needed)
    main_log_file, main_logger = setup_logging(
        log_dir=args.log_dir,
        log_level=getattr(logging, args.log_level.upper()),
        experiment_name="main_runner" # General log for the runner itself
    )
    main_logger.info(f"Main runner started. Log level: {args.log_level}")
    main_logger.info(f"Loading configuration from: {args.config}")

    try:
        config_data = load_config(args.config, logger=main_logger)
    except Exception as e:
        main_logger.critical(f"Failed to load or parse configuration file: {e}", exc_info=True)
        return
     # ---- ADD THIS CHECK ----
    if config_data is None:
        main_logger.critical(f"Configuration file '{args.config}' loaded as None. This usually means the file is empty or contains only 'null'. Please check the file content.")
        return
    # ---- END OF ADDED CHECK ----
    # Ensure the main results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a timestamped subdirectory for this entire run of main.py (optional)
    # E.g., results/run_20230101_120000/
    # This helps group all outputs from one execution of main.py if it runs multiple experiment suites.
    # For simplicity, ExperimentRunner will create its own subdirs based on experiment name and trial.
    # So, args.results_dir will be the parent for all those.

    runner = ExperimentRunner(config=config_data, results_base_dir=args.results_dir)
    
    main_logger.info("Starting all experiments defined in the configuration...")
    all_run_metrics = runner.run_all_experiments()
    main_logger.info("All experiments complete.")

    # --- Optional: Post-processing and Comparative Analysis ---
    # Example: Collect data for speedup plots if multiple runs of the same problem
    # with varying parallelism were performed.
    
    # Save a summary of all runs from this main execution
    summary_all_runs_file = os.path.join(args.results_dir, f"all_runs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(summary_all_runs_file, 'w') as f:
            json.dump(all_run_metrics, f, indent=4, default=str) # Use default=str for numpy types
        main_logger.info(f"Summary of all experiment metrics saved to: {summary_all_runs_file}")
    except Exception as e:
        main_logger.error(f"Could not save summary of all runs: {e}", exc_info=True)

    # Example logic for generating speedup plots if data is available and structured appropriately:
    # This would require parsing `all_run_metrics` to find serial baselines and corresponding parallel runs.
    # For example:
    # baseline_runtimes = {} # problem_unique_id -> runtime
    # parallel_runs = [] # list of {"problem_unique_id": ..., "num_workers_or_grid_size": ..., "runtime_seconds": ...}
    # for m in all_run_metrics:
    #     if m["status"] == "completed":
    #         # Construct a unique problem identifier (e.g., problemName_dimX)
    #         problem_id = f"{m['problem_name']}_dim{m['problem_dimensions']}"
    #         if m["model_type"] == "serial_ga": # Or your designated baseline model
    #             baseline_runtimes[problem_id] = m["runtime_seconds"]
    #         elif m["model_type"] == "master_slave": # Or other parallel models
    #            if isinstance(m["num_workers_or_grid_size"], (int, float)) and m["num_workers_or_grid_size"] > 0: # Check if it's a number
    #                 parallel_runs.append({
    #                     "problem_name": problem_id,
    #                     "num_workers_or_grid_size": m["num_workers_or_grid_size"],
    #                     "runtime_seconds": m["runtime_seconds"]
    #                 })
    # if baseline_runtimes and parallel_runs:
    #     main_logger.info("Attempting to generate speedup and efficiency plots...")
    #     comparative_plot_dir = os.path.join(args.results_dir, "comparative_analysis_plots")
    #     os.makedirs(comparative_plot_dir, exist_ok=True)
    #     plot_speedup_efficiency(
    #         results_dir=comparative_plot_dir,
    #         experiment_group_name="Overall_Comparison", # Or be more specific
    #         baseline_problem_runtime_map=baseline_runtimes,
    #         parallel_runs_data=parallel_runs
    #     )
    # else:
    #     main_logger.info("Not enough data for comparative speedup/efficiency plots (need serial and parallel runs for same problems).")

    main_logger.info(f"Main runner finished. Check '{args.results_dir}' for outputs and '{main_log_file}' for main log.")

if __name__ == "__main__":
    # This ensures that if 'main.py' is run as a script,
    # the 'ga' package can be found if 'main.py' is in the parent directory of 'ga'.
    # (Common when running from the project root: python main.py ...)
    # No explicit sys.path manipulation needed if project structure is standard.
    main()