


"""
"""
"""
Input: 
    float | jnp.ndarray

NB: jnp.ndarray could have been derived from interval or np.ndarray(dtype=interval)

1.  make_jaxpr
    1.1 args -> tree_flatten() -> args_flat
    1.2 args_flat -> to_avlas() -> args_flat_abs
    1.3 f -> flatten_fun_args() -> f_flat
    1.4 f_flat, args_flat_abs -> trace_jaxpr() -> jaxpr
    
    
"""