# ga/utils.py
import logging
import time
import json
import csv
import os
import yaml # requires PyYAML
from datetime import datetime

def setup_logging(log_dir="logs", log_level=logging.INFO, experiment_name="ga_experiment"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Incorporate experiment_name into log filename for better organization
    log_filename = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Remove existing handlers if any, to prevent duplicate logging if called multiple times
    # This is important if setup_logging might be invoked by different modules/parts of an experiment.
    # root_logger = logging.getLogger()
    # if root_logger.hasHandlers():
    #     root_logger.handlers.clear()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s", # Added logger name
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # mode='w' to overwrite for new experiment log
            logging.StreamHandler() # Also print to console
        ]
    )
    # Get a logger specific to this application/module
    logger = logging.getLogger("ParallelGA") # Or any name you prefer
    logger.info(f"Logging setup complete. Log file: {log_filename}")
    return log_filename, logger # Return logger instance too

class Timer:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger("Timer")
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        self.logger.info(f"Block executed in: {self.elapsed_time:.4f} seconds")


def save_results(results_dir: str, experiment_name: str, metrics: dict, config: dict, 
                 best_fitness_log: list, avg_fitness_log: list, logger=None):
    if logger is None: logger = logging.getLogger("SaveResults")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize experiment_name for directory creation if it contains special characters
    safe_experiment_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in experiment_name)
    experiment_path = os.path.join(results_dir, f"{safe_experiment_name}_{timestamp}")
    
    try:
        os.makedirs(experiment_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating results directory {experiment_path}: {e}")
        return None # Indicate failure

    # Save summary metrics (JSON)
    metrics_summary_file = os.path.join(experiment_path, "summary_metrics.json")
    try:
        with open(metrics_summary_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str) # default=str for non-serializable like numpy types
        logger.info(f"Summary metrics saved to {metrics_summary_file}")
    except IOError as e:
        logger.error(f"Error saving summary metrics to {metrics_summary_file}: {e}")

    # Save configuration used (JSON)
    config_file_path = os.path.join(experiment_path, "config_used.json")
    try:
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=4, default=str)
        logger.info(f"Experiment configuration saved to {config_file_path}")
    except IOError as e:
        logger.error(f"Error saving configuration to {config_file_path}: {e}")


    # Save fitness logs (CSV)
    fitness_log_file = os.path.join(experiment_path, "fitness_log.csv")
    try:
        with open(fitness_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "BestFitness", "AverageFitness"])
            max_len = max(len(best_fitness_log), len(avg_fitness_log) if avg_fitness_log else 0)
            for i in range(max_len):
                gen = i
                best_f = best_fitness_log[i] if i < len(best_fitness_log) else None
                avg_f = avg_fitness_log[i] if avg_fitness_log and i < len(avg_fitness_log) else None
                writer.writerow([gen, best_f, avg_f])
        logger.info(f"Fitness logs saved to {fitness_log_file}")
    except IOError as e:
        logger.error(f"Error saving fitness log to {fitness_log_file}: {e}")
    
    return experiment_path # Return path for plotting or further use


def load_config(config_path: str, logger=None) -> dict:
    if logger is None: logger = logging.getLogger("LoadConfig")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}. Use YAML or JSON.")
                raise ValueError("Unsupported configuration file format.")
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration from {config_path}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration from {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise


def get_cpu_gpu_utilization(logger=None):
    if logger is None: logger = logging.getLogger("Utilization")
    
    cpu_util, gpu_util_val = "N/A", "N/A"
    
    try:
        import psutil
        cpu_util = psutil.cpu_percent(interval=0.1) # Non-blocking, get overall CPU %
    except ImportError:
        logger.debug("psutil not found, CPU utilization monitoring disabled.")
    except Exception as e:
        logger.warning(f"Error getting CPU utilization with psutil: {e}")

    try:
        import pynvml
        pynvml.nvmlInit()
        # Assuming one GPU, or get utilization for a specific one.
        # Handle cases with no NVIDIA GPU or NVML issues.
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # GPU 0
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util_val = util.gpu # GPU core utilization
        else:
            gpu_util_val = "No NVIDIA GPU"
        pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not found, NVIDIA GPU utilization monitoring disabled.")
    except pynvml.NVMLError as e:
        logger.warning(f"NVML error getting GPU utilization: {e}. (Often means no NVIDIA GPU or driver issue)")
        gpu_util_val = f"NVML Error: {e}"
    except Exception as e:
        logger.warning(f"Unexpected error getting GPU utilization: {e}")


    return {"cpu_percent": cpu_util, "gpu_percent_util": gpu_util_val}