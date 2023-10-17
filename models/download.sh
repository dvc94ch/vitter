#!/bin/sh
wget https://huggingface.co/Xenova/vit-gpt2-image-captioning/raw/main/tokenizer.json
wget https://huggingface.co/Xenova/vit-gpt2-image-captioning/resolve/main/onnx/encoder_model_quantized.onnx
wget https://huggingface.co/Xenova/vit-gpt2-image-captioning/resolve/main/onnx/decoder_model_merged_quantized.onnx