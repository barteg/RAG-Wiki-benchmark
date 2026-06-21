#!/bin/bash
#SBATCH --job-name=wiki_benchmark
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# 1. Load Modules (Athena specific)
module load Python/3.10.4
module load SQLite/3.38.3

# 2. Setup Virtual Env
python -m venv venv_athena
source venv_athena/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Use Configuration from .env
# No overrides here to allow .env to take precedence
# (ensure LLM_API_KEY and LLM_BASE_URL are in .env)

# 4. Create necessary dirs
mkdir -p data/03_rag_db data/04_wiki benchmarks logs

# 4.5 Fast test limit
export BENCHMARK_LIMIT=0

# 5. Run Ingestion (Wiki Compilation)
echo "Starting Ingestion..."
python -u main.py

# 6. Run Advanced Evaluation
echo "Starting Advanced Evaluation (n=120)..."
export MAX_CONCURRENT_TASKS=8
python -u run_advanced_eval.py

echo "Benchmark Complete."
