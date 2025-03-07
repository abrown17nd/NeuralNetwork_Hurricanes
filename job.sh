#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -N pc2
#$ -M abrown17@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts

# Load conda module
source activate project_environment


# Change directory to where the Python script is located


python3 LSTM_Model_dropsondes_v1.py
