# Parallel Genetic Algorithms

This project implements two parallel Genetic Algorithm paradigms:
1. **Master-Slave (Global) Model** - Utilizing multiprocessing for distributed fitness evaluations
2. **Cellular (Fine-Grained) Model** - Leveraging GPU parallelism via CUDA

## Features

- Modular implementation of GA components (population, individuals, operators)
- Support for continuous optimization benchmarks (Griewank, Rastrigin, Sphere, etc.)
- Support for combinatorial problems (TSP, Royal-Road, etc.)
- Automatic experiment execution and result collection
- Performance visualization (convergence plots, speedup curves)
- Resource utilization monitoring

## Installation

```bash
# Clone the repository

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### CUDA Requirements

For the Cellular GA model, you need:
- NVIDIA GPU with CUDA support
- CUDA Toolkit (10.1 or later)
- PyCUDA properly installed

## Usage

### Running an experiment

```bash
# Run with default parameters
python main.py

# Specify a configuration file
python main.py --config config_examples/master_slave_experiment.yaml

# Run a specific benchmark with the Master-Slave model
python main.py --model master_slave --benchmark rastrigin --pop_size 1000 --num_workers 8

# Run with the Cellular model on GPU
python main.py --model cellular --benchmark sphere --pop_size 1024 --grid_size 32x32
```

### Configuration File Format

Configuration files use YAML format:

```yaml
model: master_slave  # or cellular
benchmark:
  name: rastrigin
  dimensions: 30
  bounds: [-5.12, 5.12]
population:
  size: 1000
  representation: real  # or binary
ga_params:
  generations: 100
  mutation_rate: 0.01
  crossover_rate: 0.8
  selection: tournament  # or roulette, rank
  tournament_size: 3     # if using tournament selection
  elitism: 2             # number of elite individuals
parallel_params:
  master_slave:
    num_workers: 8
  cellular:
    grid_size: [32, 32]
    neighborhood: moore  # or von_neumann
experiment:
  runs: 30
  seed: 42
  output_dir: results
```

## Benchmarks

The system includes the following benchmark problems:

### Continuous Optimization
- Sphere
- Rastrigin
- Griewank
- Rosenbrock
- Deceptive
- Two-peaks

### Combinatorial Problems
- Traveling Salesman Problem (TSP)
- Royal Road
- MaxBit

## Output

Results are saved in the `results/` directory with a timestamp:

```
results/2025-05-09_123456/
├── config.yaml           # Copy of the experiment configuration
├── metrics.csv           # Performance metrics for all runs
├── convergence.png       # Convergence plot
├── speedup.png           # Speedup curve
├── efficiency.png        # Parallel efficiency plot
├── resource_usage.csv    # CPU/GPU utilization data
└── logs/                 # Detailed logs for each run
```

## Developer Guide

See the documentation in the code for detailed information on extending the framework with new:
- Benchmarks
- Selection methods
- Crossover operators
- Mutation operators
- Parallelization strategies

## Running Tests

```bash
pytest
```

## License

MIT License