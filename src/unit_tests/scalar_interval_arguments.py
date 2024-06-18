
import jax
from typing import Sequence
from src.custom_interpreter.transform import interval_transformation


def f_scalar_valued(x1: float, x2: float, x3: float) -> float:
    return x1*x1*x2 + 5*x1*x2*x3 + x1*x3*x3

def f_vector_valued(x1: float, x2: float, x3: float) -> Sequence[float]:
    y1 = x1*x1*x2 + 5*x1*x2*x3 + x1*x3*x3
    y2 = x1*x1*x3 - x2*x2*x3 + x1*x2*x3
    return y1, y2

def K_primals(scalar_f, *scalarArgs, ivalArgs=None):
    return interval_transformation(scalar_f)(*scalarArgs, intervalArgs=ivalArgs)[0]

def K_adjoints(scalar_f, *scalarArgs, ivalArgs=None, wrt=0):
    scalar_grad = jax.grad(scalar_f, wrt)
    return interval_transformation(scalar_grad)(*scalarArgs, intervalArgs=ivalArgs)[0]

def K_vjp(vector_f, *scalarArgs, ivalArgs=None, seed: tuple = None):
    # todo:     the problem is that jax.vjp uses the function as well as scalarArgs or primals,
    #           consequently, the vjpFun later use the same primals and an additional contangent vector
    #           hence, we cannot replace the scalarArgs with ivalArg even with custom_interpreter.
    #           we cannot use ivalArgs as seed !
    scalarPrimalsOut, vjpFun = jax.vjp(vector_f, *scalarArgs)
    return interval_transformation(vjpFun)(seed)

adjWrt: int = 0                     # trying changing it from 0 to 2 and observe the difference.
scalarPrimalsIn: tuple = (1.0, 2.0, 3.0,)
ivalPrimalsIn: tuple = ((2.0, 3.0), (2.0, 3.0), (2.0, 3.0))
seed_m: tuple = (1.0, 0.0)           # NB: len(seed_m) = len(f(primals))

print(f"K_primals   f-scalar     =  {K_primals(f_scalar_valued, *scalarPrimalsIn, ivalArgs=ivalPrimalsIn)}")
print(f"K_adjoints  f-scalar     =  {K_adjoints(f_scalar_valued, *scalarPrimalsIn, ivalArgs=ivalPrimalsIn, wrt=adjWrt)}")
# print(f"K_vjp       f-vector     =  {K_vjp(f_vector_valued, *scalarPrimalsIn, ivalArgs=ivalPrimalsIn, seed=seed_m)}")
