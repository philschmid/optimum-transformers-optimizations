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

Micromamba
```bash
sudo apt-get install bzip2
# Assuming linux
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest -o test | tar -xvj -C ~/
~/bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# Installing packages is mostly similar
micromamba activate
micromamba install python=3.9 jupyter -c conda-forge
```

### Install python dependencies

```bash
pip install -r requirements.txt
```


## Run static quantziation with HPO 

```bash
python scripts/run_static_quantizatio_hpo.py  --model_id optimum/distilbert-base-uncased-finetuned-banking77 
```
Best result is `{'percentile': 99.99239080907178}. Best is trial 50 with value: 0.9224`
