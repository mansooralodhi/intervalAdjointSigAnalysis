import jax


class Data(object):
    def __init__(self, num_input_variables: int):
        self.key = jax.random.PRNGKey(0)
        self.no_of_variables = num_input_variables

    def generate_X(self, batch_size: int = 1) -> jax.typing.ArrayLike:
        if batch_size == 1:
            return jax.random.uniform(self.key, (self.no_of_variables,))
        return jax.random.uniform(self.key, shape=(batch_size, self.no_of_variables))

    def generate_Y(self, batch_size: int = 1) -> bool | jax.typing.ArrayLike:
        if batch_size == 1:
            return bool(jax.numpy.round(jax.random.uniform(self.key, (1,))))
        return jax.numpy.asarray(list(map(bool, jax.numpy.round(jax.random.uniform(self.key, (batch_size,))))))


if __name__ == "__main__":
    data_generator = Data(4)
    print(data_generator.generate_X(1))
    print(data_generator.generate_Y(3))
