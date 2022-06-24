from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "optimum/distilbert-base-uncased-finetuned-banking77"
dataset_id = "banking77"
dynamic_onnx_path = Path("dynamic_onnx")

# load vanilla transformers and convert to onnx
model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(dynamic_onnx_path)
tokenizer.save_pretrained(dynamic_onnx_path)

# create ORTOptimizer and define optimization configuration
dynamic_optimizer = ORTOptimizer.from_pretrained(model_id, feature=model.pipeline_task)
dynamic_optimization_config = OptimizationConfig(optimization_level=99)  # enable all optimizations

# apply the optimization configuration to the model
dynamic_optimizer.export(
    onnx_model_path=onnx_path / "model.onnx",
    onnx_optimized_model_output_path=dynamic_onnx_path / "model-optimized.onnx",
    optimization_config=dynamic_optimization_config,
)

# create ORTQuantizer and define quantization configuration
dynamic_quantizer = ORTQuantizer.from_pretrained(model_id, feature=model.pipeline_task)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# apply the quantization configuration to the model
dynamic_quantizer.export(
    onnx_model_path=dynamic_onnx_path / "model-optimized.onnx",
    onnx_quantized_model_output_path=dynamic_onnx_path / "model-quantized.onnx",
    quantization_config=dqconfig,
)

import os

# get model file size
size = os.path.getsize(dynamic_onnx_path / "model.onnx") / (1024 * 1024)
print(f"Vanilla Onnx Model file size: {size:.2f} MB")
size = os.path.getsize(dynamic_onnx_path / "model-quantized.onnx") / (1024 * 1024)
print(f"Quantized Onnx Model file size: {size:.2f} MB")

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(dynamic_onnx_path, file_name="model-quantized.onnx")

dynamic_clx = pipeline("text-classification", model=model, tokenizer=dynamic_quantizer.tokenizer)

from evaluate import evaluator
from datasets import load_dataset

eval = evaluator("text-classification")
eval_dataset = load_dataset("banking77", split="test")

results = eval.compute(
    model_or_pipeline=dynamic_clx,
    data=eval_dataset,
    metric="accuracy",
    input_column="text",
    label_column="label",
    label_mapping=model.config.label2id,
    strategy="simple",
)
print(results)
