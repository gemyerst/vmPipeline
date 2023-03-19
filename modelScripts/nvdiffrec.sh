#!/bin/bash

docker run --gpus all -v $1:/mnt/ -w /mnt/ nvdiffrec:latest /bin/bash -c "git clone https://github.com/gemyerst/nvdiffrec.git nvdiffrec_eval \
    && cd nvdiffrec_eval \
    && python train.py --config /mnt/nvdiffrec/config.json"
