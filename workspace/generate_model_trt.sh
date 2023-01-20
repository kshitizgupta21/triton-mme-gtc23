#!/bin/bash

echo "Installing Transformers..."
pip -q install transformers[onnx]

echo "Exporting model to ONNX..."
python -m transformers.onnx --model=bergum/xtremedistil-emotion \
                            --feature=sequence-classification /workspace/onnx/

# use CUDA LAZY LOADING
export CUDA_MODULE_LOADING=LAZY
echo "Converting ONNX Model to TensorRT FP16 Plan..."
trtexec --onnx=/workspace/onnx/model.onnx \
        --saveEngine=/workspace/model.plan \
        --minShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
        --optShapes=input_ids:16x128,attention_mask:16x128,token_type_ids:16x128 \
        --maxShapes=input_ids:224x128,attention_mask:224x128,token_type_ids:224x128 \
        --fp16 \
        --verbose \
        --memPoolSize=workspace:14000 | tee conversion_trt.txt

echo "Finished exporting all models..."
