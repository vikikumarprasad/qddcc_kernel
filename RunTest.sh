#!/bin/bash
#SBATCH --account=def-vikikrpd  # Specify the allocation account (check your allocations on CCDB)
#SBATCH -N 1                    # Number of compute nodes requested
#SBATCH -n 1                    # Number of CPU cores requested (adjust as needed)
# --mem-per-cpu=16G      # Memory per CPU core (e.g., 512M, 1G, 4G) (Don't use this on Niagara)
# --time=00-18:00:00      # Maximum wall time (D-HH:MM:SS format) - JOB WILL BE KILLED AFTER THIS
# --job-name=Data_"$STRUCTURE""$FILE_NUM"_"BASIS" # A descriptive name for your job
#SBATCH --output=logs/%x.out  # Standard output file (%x=jobname, %j=jobid) ('/path/to/your/project/logs' has to already exist)
#SBATCH --error=logs/%x.err   # Standard error file (%x=jobname, %j=jobid)
#SBATCH --mail-user=gage.holmen@gmail.com # Email address for notifications (optional)
#SBATCH --mail-type=FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT,END # Send email on job failure or other events

echo "Job started at: $(date)"
# # echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# echo "--- Environment Setup ---"
module load apptainer

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# Install dependencies (use --no-index if you have pre-downloaded wheels)
# See: https://docs.alliancecan.ca/wiki/Python#Installing_packages_in_a_virtual_environment
# pip install --no-index --upgrade pip

# pip install --no-index -r /path/to/your/project/requirements.txt

echo "--- Running Python Script ---"

# --- Configuration ---
# Update these variables for your specific setup
# Make sure your Apptainer image, Python script, and any input files are in the same directory
APPTAINER_IMAGE="qml_research.sif"             # Your Apptainer image file, in the same directory
CONDA_ENV_NAME="qml_research"                 # The name of your Conda environment inside the image
PYTHON_SCRIPT="QKernel_Methods_Test.py"         # Your Python script, in the same directory
# --- End Configuration ---

#apptainer exec qml_research.sif python QKernel_Methods_Test.py
#python QKernel_Methods_Test.py

#apptainer exec "${APPTAINER_IMAGE}" bash -c "source /opt/conda/bin/activate ${CONDA_ENV_NAME} && python ${PYTHON_SCRIPT}"
apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    "${APPTAINER_IMAGE}" \
    bash -c "source /opt/conda/bin/activate ${CONDA_ENV_NAME} && python ${PYTHON_SCRIPT}"

EXIT_CODE=$?
echo "--- Python script finished with exit code $EXIT_CODE ---"

# Optional: Copy important results from $SLURM_TMPDIR back to persistent storage if needed
# cp $SLURM_TMPDIR/some_temp_output.dat /path/to/your/project/results/

echo "Job finished at: $(date)"
exit $EXIT_CODE
