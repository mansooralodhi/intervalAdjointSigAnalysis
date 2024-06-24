

"""
Usage:
    - import src.custom_interpreter.transformation import  scalar_primal_transformation,
                                                    interval_primal_transformation,
                                                    scalar_adjoint_transformation,
                                                    interval_adjoint_transformation,

Interpreter Features:
    - computations
        - scalar-valued function primal computation
        - scalar-valued function adjoint computation
        - vector-valued function primal computation
        - vector-valued function adjoint computation
        - vector-valued function vjp computation  [Tricky]

    - custom_vjp calls using custom custom_interpreter
    - custom_vjp_call_jaxpr calls using custom custom_interpreter
    - pjit calls using custom custom_interpreter
    - vmap calls using custom custom_interpreter
    - xla_call using custom custom_interpreter [Optional]
    - xla_pmap using custom custom_interpreter [Optional]

To Do:
    - compute vjp with the help of jax.vjp using interval args



"""