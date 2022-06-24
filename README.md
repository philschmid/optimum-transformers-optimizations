# Optimizing Transformers with Optimum

In this session, you will learn how to optimize Hugging Face Transformers models using Optimum. The session will show you how to dynamically quantize and optimize a DistilBERT model using [Hugging Face Optimum](https://huggingface.co/docs/optimum/index) and [ONNX Runtime](https://onnxruntime.ai/). Hugging Face Optimum is an extension of 🤗 Transformers, providing a set of performance optimization tools enabling maximum efficiency to train and run models on targeted hardware.

Note: dynamic quantization is currently only supported for CPUs, so we will not be utilizing GPUs / CUDA in this session.

By the end of this session, you see how quantization and optimization with Hugging Face Optimum can result in significant increase in model latency while keeping almost 100% of the full-precision model. Furthermore, you’ll see how to easily apply some advanced quantization and optimization techniques shown here so that your models take much less of an accuracy hit than they would otherwise. 

You will learn how to:
1. Setup Development Environment
2. Convert a Hugging Face `Transformers` model to ONNX for inference
3. Apply graph optimization techniques to the ONNX model
4. Apply dynamic quantization using ORTQuantizer from 🤗 Optimum
5. Test inference with the quantized model
6. Evaluate the performance and speed
7. Push the quantized model to the Hub
8. Load and run inference with a quantized model from the hub

Let's get started! 🚀

---

## Setup


### [Miniconda](https://waylonwalker.com/install-miniconda/#installing-miniconda-on-linux) or [Micromamba](https://labs.epi2me.io/conda-or-mamba-for-production/) setup (conda alternative but smaller)

Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
### Install python dependencies

```bash
pip install -r requirements.txt
```

# Text Classification Optimum API Template

This is a template repository for Text Classification using Optimum and onnxruntime to support generic inference with Hugging Face Hub generic Inference API. There are two required steps:

1. Specify the requirements by defining a `requirements.txt` file.
2. Implement the `pipeline.py` `__init__` and `__call__` methods. These methods are called by the Inference API. The `__init__` method should load the model and preload the optimum model and tokenizers as well as the `text-classification` pipeline needed for inference. This is only called once. The `__call__` method performs the actual inference. Make sure to follow the same input/output specifications defined in the template for the pipeline to work.

add 
```
library_name: generic
```
to the readme.

_note: the `generic` community image currently only support `inputs` as parameter and no parameter._