# Vitter

Specify input files in png/jpg/gif/webp/svg format and it will output a textual description
in an optional output directory.

## Dependencies
- libonnxruntime

## AI model
- [https://huggingface.co/Xenova/vit-gpt2-image-captioning](https://huggingface.co/Xenova/vit-gpt2-image-captioning)

Download the AI model using `models/download.sh`.

## What to expect
- installing libonnxruntime is a pain, but `tract` can't load the model currently

## License
Apache-2.0 + MIT