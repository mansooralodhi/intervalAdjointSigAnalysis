




""" """

"""
Source:
        1. https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#
        2. https://jax.readthedocs.io/en/latest/jep/2026-custom-derivatives.html

vjp:    vector-jacobian-product = adjoint-mode AD
        (unit or basis) vector = seeds the outputs
        (1 * m) . (m * n)  -> 1 * n

jvp:    jacobian-vector-product = tangent-mode AD
        (unit or basis) vector = seeds the inputs
        (m * n) . (n * 1)  -> m * 1

"""
