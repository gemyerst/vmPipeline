#!/bin/bash

docker run --gpus all -v $1:/mnt/ -w /mnt/ visionnerf:latest -c " git clone https://github.com/gemyerst/vision-nerf.git visionnerf_eval \
    && cd visionnerf_eval \
    && gcloud storage cp gs://vision-nerf-weights/srn_cars_500000.pth /mnt/visionnerf/weights/srn_cars_500000.pth \
    && python eval.py --config /mnt/visionnerf/config.txt --data_path /mnt/visionnerf/data"
