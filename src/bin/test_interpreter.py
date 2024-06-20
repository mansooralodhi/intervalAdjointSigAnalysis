import jax
import jax.numpy as jnp
from jax import random as rnd
from src.interpreter.interpreter import Interpreter

def vjp(x):
    primal, model_vjp = jax.vjp(model, x)
    grad_x = model_vjp(jnp.ones(shape=(1,)))
    return grad_x

class Model():
    def __init__(self):
        self.w1 = rnd.normal(rnd.key(0), shape=(264, 128))
        self.w2 = rnd.normal(rnd.key(0), shape=(128, 1))

    def __call__(self, x):
        y = jnp.dot(jnp.dot(x, self.w1), self.w2)[0]
        return y


if __name__ == "__main__":
    model = Model()
    interpreter = Interpreter()

    x = rnd.normal(rnd.key(0), shape=(264,))
    print(f"Primal = {model(x)}")

    # jaxepr = jax.make_jaxpr(model)(x)
    # y = interpreter.safe_interpret(jaxepr.jaxpr, jaxepr.literals, [x])[0]
    # print(f"K_primal = {y}")
    #
    # jaxepr = jax.make_jaxpr(jax.grad(model))(x)
    # y = interpreter.safe_interpret(jaxepr.jaxpr, jaxepr.literals, [x])[0]
    # print(f"K_primal = {y}")


    # primal, model_vjp = jax.vjp(model, x)
    # grad_x = model_vjp(jnp.ones(shape=(1,)))[0]
    # print(f"J_adj len = {len(grad_x)}")
    #
    # jaxpr = jax.make_jaxpr(vjp)(x)
    # y = interpreter.safe_interpret(jaxpr.jaxpr, jaxpr.literals, [x])[0]
    # print(f"K_adj len = {len(y)}")
    #
    # print(f"K_adj = J_adj ?  {all(y == grad_x)}")
