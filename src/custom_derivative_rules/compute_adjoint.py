

import jax
from src.custom_interpreter.interpreter import Interpreter
from src.custom_derivative_rules.standalone import x_active, loss
from src.custom_interpreter.transform import transformation, interval_transformation

interpret = Interpreter()

def J_primal(f, x):
    return f(x)

def K_primal(f, x):
    return transformation(f)(x)

def J_adjoint(f, x):
    return jax.grad(f)(x)

def K_adjointl(f, x):
    return transformation(jax.grad(f))(x)


if __name__ == "__main__":
    w = x_active()
    # print(J_primal(loss, w))
    # print(K_primal(loss, w))
    # y = J_adjoint(loss, w)
    y = K_adjointl(loss, [w])
    print(y)
