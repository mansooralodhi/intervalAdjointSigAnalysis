import onnx, jax
import numpy as np
from jaxonnxruntime import backend
"""i need to import JaxBackend to make this library work"""
import jaxonnxruntime as jort
from intervalArithmetic import IntervalArithmetic

def run(model):
    model = onnx.load(model)
    onnx.checker.check_model(model)
    dummy_data = np.random.rand(1, 8)
    dummy_data = jax.random.uniform(jax.random.PRNGKey(0), (8,))
    dummy_data = IntervalArithmetic(1.0, 1.0)
    jax_func, model_params = jort.call_onnx.call_onnx_model(model, [dummy_data])
    print(jax_func(model_params, [dummy_data]))
    jax_onnx_model = lambda x: jax_func(model_params, [dummy_data])[0].squeeze()
    jax_jit_pruned_model = jax.jit(jax.vmap(jax_onnx_model))
    print(jax_jit_pruned_model(dummy_data))


if __name__ == "__main__":
    model = "../primary_model/artifacts/non_linear_regression_model.onnx"
    run(model)
