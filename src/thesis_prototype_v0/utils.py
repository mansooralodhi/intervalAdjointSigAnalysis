import jax


def vector_value_function(x: jax.Array) -> jax.Array:
    """
    f : R^4 -> R^3
    inputs:
        x -> shape (4,)
    output:
        y -> shape (3,)
    """
    if len(x) != 4:
        raise Exception("Invalid Input Length !")
    u = jax.numpy.asarray([
        x[0] ** 6 * x[1] ** 4 * x[2] ** 9 * x[3] ** 2,
        x[0] ** 2 * x[1] ** 3 * x[2] ** 5 * x[3] ** 3,
        x[0] ** 5 * x[1] ** 7 * x[2] ** 7 * x[3] ** 6,
    ])
    return u


if __name__ == "__main__":
    z = jax.numpy.asarray([1.0, 0.5, 1.5, 2.0])
    print(vector_value_function(z))
