#!/bin/bash

docker run --gpus all -v $1:/mnt/vmPipeline -w /mnt/vmPipeline vision-nerf:latest /bin/bash -c "git clone https://github.com/gemyerst/vision-nerf.git visionnerf_eval && cd visionnerf_eval && python eval.py --config /mnt/vmPipeline/visionnerf/config.txt"
