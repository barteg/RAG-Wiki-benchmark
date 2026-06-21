#!/bin/bash
# Athena Submission Helper
# Usage: ./athena_run.sh <plgrid_username>

INPUT_USER=$1
if [ -z "$INPUT_USER" ]; then
    echo "Usage: ./athena_run.sh <your_plgrid_username>"
    exit 1
fi

# Extract local username if full email provided
USERNAME=$(echo $INPUT_USER | cut -d'@' -f1)
ATHENA_HOST="athena.cyfronet.pl"
REMOTE_DIR="~/wiki_benchmark"

echo "🚀 Packaging project..."
tar -czf wiki_benchmark.tar.gz --exclude='venv' --exclude='venv_athena' --exclude='__pycache__' --exclude='.git' .

echo "📦 Transferring to Athena for user: ${USERNAME}..."
ssh ${USERNAME}@${ATHENA_HOST} "mkdir -p ${REMOTE_DIR}"
scp wiki_benchmark.tar.gz ${USERNAME}@${ATHENA_HOST}:${REMOTE_DIR}/

echo "🔧 Unpacking on Athena..."
ssh ${USERNAME}@${ATHENA_HOST} "cd ${REMOTE_DIR} && tar -xzf wiki_benchmark.tar.gz && rm wiki_benchmark.tar.gz"

echo "📝 Submitting SLURM job to 'plgrid' partition..."
ssh ${USERNAME}@${ATHENA_HOST} "cd ${REMOTE_DIR} && sbatch benchmark.slurm"

rm wiki_benchmark.tar.gz

echo "📊 Monitor job: ssh ${USERNAME}@${ATHENA_HOST} 'squeue -u ${USERNAME}'"
echo "📜 View output: ssh ${USERNAME}@${ATHENA_HOST} 'tail -f ${REMOTE_DIR}/logs/benchmark_*.out'"
