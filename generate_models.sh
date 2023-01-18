#!/bin/bash

docker run --gpus=all --rm -it \
            -v `pwd`:/workspace \
             --ulimit memlock=-1 --ulimit stack=67108864 \
             nvcr.io/nvidia/pytorch:22.04-py3 \
             /bin/bash export_models.sh
             
mkdir -p model_repository/xdistilbert_pt/1
mkdir -p model_repository/xdistilbert_trt/1

mv model.pt model_repository/xdistilbert_pt/1/
mv model.plan model_repository/xdistilbert_trt/1/

echo "Finished generating all models."
