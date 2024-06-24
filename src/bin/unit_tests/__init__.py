

"""
Unit test following scenarios with custom custom_interpreter:
-----------------------------------------------------
    - scalar and interval args:
        - scalar-valued function adjoint computation
        - vector-valued function adjoint computation  [Tricky]
        - custom_vjp calls using custom custom_interpreter
        - custom_vjp_call_jaxpr calls using custom custom_interpreter
    - pjit calls using custom custom_interpreter
    - vmap calls using custom custom_interpreter
    - xla_call using custom custom_interpreter [Optional]
    - xla_pmap using custom custom_interpreter [Optional]

Test cases template:
-------------------
    - Inputs: Multivariate Scalar/Interval
    - Outputs: Scalar/Interval
    - Function: Scalar-Valued / Vector-Valued
    - Routines:
        - J -> Jax Interpreter
        - K -> Custom Interpreter
        - J_primals / J_adjoints (scalar-f) / J_vjp (vector-f)
        - K_primals / K_adjoints (scalar-f) / K_vjp (vector-f)
    - Limitation: only active parameters
    - Contangent Vector: refers to the unit vector of size m used during reverse-pass (seed)

Jax.vjp:
--------
    - primals, vjp_fun = jax.vjp(f, *args)
    - primals = f(*args)
    - contangent_vector = (1.0, 0.0, ...)         #   len(primals) == len(contangent_vector)
    - args_adjoints = vjp_fun(contangent_vector)  #   len(args_adjoints) == len(args)

"""