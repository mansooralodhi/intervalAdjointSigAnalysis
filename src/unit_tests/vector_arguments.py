
import jax
from typing import Sequence
from src.custom_interpreter.interpreter import Interpreter


interpreter = Interpreter()

def f_scalar_valued(x_active: Sequence[float], x_passive: Sequence[float]) -> float:
    return x_active[0]*x_passive[0]*x_active[1] + 5*x_active[0]*x_passive[1]*x_active[1] + x_passive[0]*x_passive[1]*x_active[2]

def f_vector_valued(x_active: Sequence[float], x_passive: Sequence[float]) -> Sequence[float]:
    y1 = x_active[0]*x_passive[0]*x_active[1] + 5*x_active[0]*x_passive[1]*x_active[1] + x_passive[0]*x_passive[1]*x_active[2]
    y2 = x[0]*x[0]*x[2] - x[1]*x[1]*x[2] + x[0]*x[1]*x[2]
    return y1, y2

def J_primals(scalar_f, *args):
    return scalar_f(*args)

def J_adjoints(scalar_f, *args, wrt=0):
    return jax.grad(scalar_f, wrt)(*args)

def J_vjp(scalar_f, *args, seed: tuple = None):
    primals_out, vjp_fun = jax.vjp(scalar_f, *args)
    return vjp_fun(seed)

def K_primals(scalar_f, args):
    expr = jax.make_jaxpr(scalar_f)(args)
    return interpreter.safe_interpret(expr.jaxpr, expr.literals, [*args])[0]

def K_adjoints(scalar_f, args, wrt: int = 0):
    expr = jax.make_jaxpr(jax.grad(scalar_f, wrt))(args)
    return interpreter.safe_interpret(expr.jaxpr, expr.literals, [*args])[0]

def K_vjp(scalar_f, *args, seed: tuple = None):
    primals_out, vjp_fun = jax.vjp(scalar_f, *args)
    expr = jax.make_jaxpr(vjp_fun)(seed)
    return interpreter.safe_interpret(expr.jaxpr, expr.literals, seed)


adjWrt: int = 0                     # trying changing it from 0 to 2 and observe the difference.
primalsIn: tuple = ((3.0, 4.0), (5.0), )
seed_m: tuple = (1.0, 0.0)           # NB: len(seed_m) = len(f(primals))

print(f"J_primals   f-scalar     =  {J_primals(f_scalar_valued, primalsIn)}")
print(f"J_adjoints  f-scalar     =  {J_adjoints(f_scalar_valued, primalsIn, wrt=adjWrt)}")
print(f"J_vjp       f-vector     =  {J_vjp(f_vector_valued, primalsIn, seed=seed_m)}")
print(f"K_primals   f-scalar     =  {K_primals(f_scalar_valued, primalsIn)}")
print(f"K_adjoint   f-scalar     =  {K_adjoints(f_scalar_valued, primalsIn, wrt=adjWrt)}")
print(f"K_vjp       f-vector     =  {K_vjp(f_vector_valued, primalsIn, seed=seed_m)}")


