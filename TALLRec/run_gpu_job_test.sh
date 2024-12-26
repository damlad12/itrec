#!/bin/bash
#$ -N gpu_evaluate_job     # Job name
#$ -l gpus=1               # Request 1 GPU
#$ -l gpu_memory=80G       # Request GPUs with at least 80 GB of memory
#$ -l gpu_c=3.5            # Request GPUs with compute capability 3.5 or higher
#$ -l h_rt=01:00:00        # Set runtime limit
#$ -m b                    # Send an email when the job begins
#$ -m e                    # Send an email when the job ends
#$ -o eval_output.log      # Output log file
#$ -e eval_error.log       # Error log file
#$ -j y                    # Merge output and error logs
#$ -cwd                    # Run job in the current working directory

# Set up environment
export PATH=/share/pkg.7/python3/3.8.10/install/bin:$PATH
export LD_LIBRARY_PATH=/share/pkg.7/python3/3.8.10/install/lib:$LD_LIBRARY_PATH
source /projectnb/ec523kb/projects/teams_Fall_2024/Team_8/TALLRec/myenv/bin/activate

# Assign the first visible GPU ID to the job
ASSIGNED_GPU_PHYSICAL_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Assigned GPU Physical ID: $ASSIGNED_GPU_PHYSICAL_ID"

# Arguments for evaluate.sh
# OUTPUT_DIR="/projectnb/ec523kb/projects/teams_Fall_2024/Team_8/book_42_64"  # Update with the actual directory to evaluate
OUTPUT_DIR="/content/drive/MyDrive/EC523DL/Final_project/book_42_64"
chmod +x ./shell/evaluate.sh

# Execute evaluate.sh with GPU ID and output directory as arguments
./shell/evaluate.sh $ASSIGNED_GPU_PHYSICAL_ID $OUTPUT_DIR

# Deactivate the virtual environment after completion
deactivate
