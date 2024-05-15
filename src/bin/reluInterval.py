
import jax
from flax import linen
from src.interpreter.interpret import safe_interpret


def model(x, W, b):
    return jax.numpy.dot(linen.relu((jax.numpy.dot(x, W[0]) + b[0])), W[1]) + b[1]

def initialize_wrights():
    key = jax.random.key(0)
    keys = jax.random.split(key)
    W = jax.random.normal(keys[0], (5, 1))
    b = jax.random.normal(keys[1], (1,))
    return dict(W=W, b=b)

def inputX():
    key = jax.random.key(0)
    return jax.random.normal(key, (5,))

def intervalX(x):
    lb = x - 0.5
    ub = x + 0.5
    return (lb, ub)

def get_primal(x, ivalX):
    weights = initialize_wrights()
    expr = jax.make_jaxpr(model)(x, weights['W'], weights['b'])
    y = safe_interpret(expr.jaxpr, expr.literals, [ivalX, weights['W'], weights['b']])[0]
    t = model(x, weights['W'], weights['b'])
    return y, t


def get_adjoint(x, ivalX):
    weights = initialize_wrights()
    wrapper = jax.grad(model, argnums=0)
    expr = jax.make_jaxpr(wrapper)(x, weights['W'], weights['b'])
    y = safe_interpret(expr.jaxpr, expr.literals, [ivalX, weights['W'], weights['b']])[0]
    t = jax.grad(model, argnums=0)(x, weights['W'], weights['b'])
    return y, t

if __name__ == "__main__":
    x = inputX()
    ivalx = intervalX(x)
    y, t = get_adjoint(x, ivalx)
    print(y)
    print(t)