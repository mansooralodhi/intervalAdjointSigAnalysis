
def interval_exp(x):
    if isinstance(x, Interval):
        pass
    elif isinstance(x, jnp.ndarray):
        if x.shape == (2,):
            return jnp.array([jnp.exp(x[0]), jnp.exp(x[1])])
        elif x.shape == ():
            return jnp.array(jnp.exp(x))
    else:
        raise NotImplementedError("interval mul not correctly implemented")

def interval_add(x, y):
    # TODO: put the real mul in here
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return x + y
    elif isinstance(x, float) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x + y.range[0], y.range[1]])
    elif isinstance(x, jnp.ndarray) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x[0] + y.range[0], x[1] + y.range[1]])
    else:
        raise NotImplementedError("interval mul not correctly implemented")


def interval_mul(x, y):
    # TODO: put the real mul in here
    if isinstance(x, float) and isinstance(y, jnp.ndarray):
        return x * y
    elif isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.asarray([x[0] * y[0] , x[1] * y[1]])
    elif isinstance(x, float) and isinstance(y, Interval):
        # return Interval(x+y.range[0], y.range[1])
        return jnp.asarray([x * y.range[0], y.range[1]])
    else:
        raise NotImplementedError("interval mul not correctly implemented")