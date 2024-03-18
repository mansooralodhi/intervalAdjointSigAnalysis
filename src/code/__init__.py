

"""
"""
"""
Algorithm
---------
    Inputs:
        float | jnp.ndarray (dtype=float) | interval | np.ndarray (dtype= interval) 
    Steps:
        1.  make_jaxpr
            1.1 args -> tree_flatten() -> args_flat
            1.2 args_flat -> to_avlas() -> args_flat_abs
            1.3 f -> flatten_fun_args() -> f_flat
            1.4 f_flat, args_flat_abs -> trace_jaxpr() -> jaxpr
        2.  interpret_jaxpr
            2.1 preliminaries:
                2.1.1 write jax primitive ops for interval inputs
                2.1.2 create registry
                2.1.3 store primitive ops in registry
            2.2 create env
            2.3 store variable values in env
            2.4 store constants values in env
            2.5 find eqn in jaxpr
                2.5.1 get values from env
                2.5.2 get method from registry 
                2.5.3 get output 
                2.5.4 store output in env
            2.6 get output from env
            2.7 return output
        3.  wrap make_jaxpr & interpret_jaxpr   
        4.  grad w.r.t args 
        5.  grad at point x
    Outputs:
        grad
 
Arguments Type
--------------

Case I:
    Input: float | jnp.ndarray 

Case II:
    Input:  interval | np.ndarray
    Transformation: 
        i.  interval -> transform() -> jnp.array(dtype=float, shape=(2,))
        ii. np.ndarray(dtype=interval, shape=(...)) -> transform() -> jnp.ndarray(dtype=float, shape=(..., 2))    

is_leaf: Number | np.ndarray | jnp.ndarray

core input dtype: float | interval
input matrices: np.ndarray | jnp.ndarray

NB: At any given instance we are either dealing with floats (jnp.ndarray)
    or with intervals (np.ndarray)


Input Array Types
-----------------

1.  Scalar
2.  Vector
3.  Matrix

"""