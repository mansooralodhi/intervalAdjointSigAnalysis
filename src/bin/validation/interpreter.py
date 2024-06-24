import jax
from src.model.runtime import ModelRuntime
from src.site_packages.custom_interpreter import safe_interpret

"""
Compute primals and adjoints with scalar inputs and compare the results with jax methods.
"""


class InterValidator(object):
    def __init__(self, model_file: str = ''):

        self.modelRuntime = ModelRuntime(model_file)
        self.params = self.modelRuntime.model_params
        self.x = self.modelRuntime.sampleX

    def verify_primals(self):
        loss = self.modelRuntime.loss(self.x, self.params)
        expr = self.modelRuntime.primal_jaxpr(self.x, self.params)
        flatParams, _ = jax.tree_flatten(self.params)
        flatParams.insert(0, self.x)
        x_est_loss = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
        if round(loss, 3) == round(x_est_loss, 3):
            print("Scalar Primals Verified !")
        else:
            raise ValueError(x_est_loss)

    def verify_param_adjoints(self):
        pass

    def verify_input_adjoints(self):
        adjoints = self.modelRuntime.grad(self.x, self.params, wrt_arg=0)
        expr = self.modelRuntime.adjoint_jaxpr(self.x, self.params, wrt_arg=0)
        flatParams, _ = jax.tree_flatten(self.params)
        flatParams.insert(0, self.x)
        x_est_adjoint = safe_interpret(expr.jaxpr, expr.literals, flatParams)[0]
        if all(adjoints == x_est_adjoint):
            print("Scalar Adjoints Verified !")
        else:
            raise ValueError()


if __name__ == '__main__':
    validate = InterValidator()
    validate.verify_primals()
    validate.verify_input_adjoints()
