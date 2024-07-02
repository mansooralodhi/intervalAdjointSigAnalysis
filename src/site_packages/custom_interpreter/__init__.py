

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


# todo: the idea we are going to implement in order to find the intermediate primals
#       is that we take an kwarg 'argnums' similar to jax.jacrev.
#       for time being we constraint the kwarg value to be 0, 1, None  which means
#       0: x
#       1: params
#       None: no intermediate outputs expected.
#       the ideology behind obtained the desired intermediate results is we
#       store the ids/keys of variables wrt their argnums. later, in eqn for-loop
#       if those variables are used, then we store the output of that equation in
#       the new envrionment. finally we return this env with intermediate-primals.
#       we assume this would ignore the intermediate functions that don't directly
#       use that variable, such as relu activation function.

"""