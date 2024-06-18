

"""

Interpreter Features:
    - scalar and interval args
        - scalar-valued function adjoint computation
        - vector-valued function adjoint computation  [Tricky]
        - custom_vjp calls using custom interpreter
        - custom_vjp_call_jaxpr calls using custom interpreter
    - pjit calls using custom interpreter
    - vmap calls using custom interpreter
    - xla_call using custom interpreter [Optional]
    - xla_pmap using custom interpreter [Optional]
To Do:
    - compute vjp with the help of jax.vjp using interval args
"""