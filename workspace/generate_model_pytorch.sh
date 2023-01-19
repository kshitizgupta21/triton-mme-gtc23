#!/bin/bash

# use this script to save xtremedistilbert model as a .pt file
echo "Installing Transformers..."
pip -q install transformers[onnx]

echo "Downloading xdistillbert model from HuggingFace..."
echo "Exporting model to Torchscript..."
python pt_exporter.py

