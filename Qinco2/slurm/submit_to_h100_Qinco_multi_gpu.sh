#!/bin/bash

## Usage:
# bash submit_to_GPU.slurm job_name "bash_scrip.sh --arg1 val" [--time=HH:MM:SS] [--test]
# 注意：[--time=HH:MM:SS]和[--test]只能选其一，--test表示跑20min （和v100的30min不一样）

# Get date
current_date=$(date +%Y%m%d)

# Log path
log_path="/lustre/fsn1/projects/rech/thj/uth68ud/slurm_out/" # 注意：结尾的"/"不能少
mkdir -p $log_path

# Required arguments
job_name="$1"
bash_script="$2"
shift 2


# Default values
custom_time=""
test_mode="false"

# Parse optional args
for arg in "$@"; do
    case $arg in
        --time=*)
            custom_time="${arg#--time=}"
            ;;
        --test)
            test_mode="true"
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Conflict check
if [[ -n $custom_time && $test_mode == "true" ]]; then
    echo "Error: Cannot use both --time and --test together."
    exit 1
fi

# Set final time
if [[ $test_mode == "true" ]]; then
    slurm_time="00:20:00"
elif [[ -n $custom_time ]]; then
    slurm_time="$custom_time"
else
    slurm_time="20:00:00"
fi

sbatch <<EOT
#!/bin/bash

# Job name
#SBATCH --job-name=${job_name}_${current_date}      # name of job

# Partition and resources based on GPU type
#SBATCH -A thj@h100
#SBATCH -C h100


# Nodes and tasks
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:4                 # number of GPU requested per node (max. 4 for H100 nodes)
# Since here only one GPU per task is requested (i.e., 1/4 of the available GPUs)
# the best way to proceed is to book 1/4 of the node's CPU for each task, which is 24:
#SBATCH --cpus-per-task=96           # number of CPU per task (1/4 of the CPUs here)

# Other Slurm directives
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=${slurm_time}              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=${log_path}slurm_log_${job_name}.log    # name of output file
#SBATCH --error=${log_path}slurm_log_${job_name}.log    # name of error file (here, in common with the output file)

module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0

# Echo of launched commands
set -x

# Code execution

#export PYTHONNOUSERSITE=0
#echo "PYTHONNOUSERSITE set to 0!"
#export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$WORK/.local/lib/python3.11/site-packages:$PYTHONPATH
#echo "PYTHON PATH UPDATED!"
#
#echo "=== WHICH PYTHON ==="
#which python

#python -c "import pytorch_adapt; print('pytorch_adapt path:', pytorch_adapt.__file__)"
#python -c "import netCDF4; print('netCDF4 path:', netCDF4.__file__)"

#echo "=== PYTHON EXECUTABLE ==="
#python -c "import sys; print(sys.executable)"
#
#echo "=== PYTHON PATH ==="
#python -c "import sys; print('\n'.join(sys.path))"

#echo "=== LIST PACKAGES ==="
#python -m pip list | grep netCDF4

${bash_script}

exit 0
EOT
