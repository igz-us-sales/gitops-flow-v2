
import mlrun

from cloudpickle import load
from typing import List
import numpy as np
import shap

class ClassifierModelSHAP(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
    
    def explain(self, body: dict) -> List:
        """Generate model explaination from sample"""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        shap_values = self.explainer.shap_values(feats)
        return shap_values[result.argmax()].tolist()
