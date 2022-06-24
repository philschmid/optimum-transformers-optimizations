from typing import  Dict, List, Any
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer


class PreTrainedPipeline():
    def __init__(self, path=""):
        # load the optimized model
        model = ORTModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        # create inference pipeline
        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)


    def __call__(self, inputs: Any) -> List[List[Dict[str, float]]]:
        """
        Args:
            data (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        # pop inputs for pipeline
        return self.pipeline(inputs)