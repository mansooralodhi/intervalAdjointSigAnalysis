
def convert_element_type(self, a: Union[NDArrayLike, IntervalLike], new_dtype, weak_type) -> int:
    if isinstance(a, tuple):
        return jax.lax.convert_element_type(a[0], new_dtype, weak_type), jax.lax.convert_element_type(a[1], new_dtype,
                                                                                                      weak_type)
    return jax.lax.convert_element_type(a, new_dtype)