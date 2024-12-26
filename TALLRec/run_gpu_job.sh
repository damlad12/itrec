#!/bin/bash
#$ -N gpu_job             #1
#$ -l gpus=1
#$ -l gpu_memory=80G      # Request GPUs with at least 80 GB of memory
#$ -l gpu_c=8.0           # Request GPUs with compute capability 8.0 or higher
#$ -l h_rt=04:00:00       
#$ -o output.log          
#$ -e error.log
#$ -m b
#$ -m e        
#$ -j y                   
#$ -cwd    

# export PATH=/share/pkg.7/python3/3.8.10/install/bin:$PATH
# export LD_LIBRARY_PATH=/share/pkg.7/python3/3.8.10/install/lib:$LD_LIBRARY_PATH

# source /content/drive/MyDrive/EC523DL/Final_project/TALLRec/myenv/bin/activate
chmod +x ./shell/instruct_7B.sh

ASSIGNED_GPU_PHYSICAL_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Assigned GPU Physical ID: $ASSIGNED_GPU_PHYSICAL_ID"

# Pass the assigned GPU ID to the training script
# Assuming instruct_7B.sh accepts GPU ID as the first argument and seed as the second
./shell/instruct_7B.sh $ASSIGNED_GPU_PHYSICAL_ID 42
# deactivate                                                                                                                                                                                        
