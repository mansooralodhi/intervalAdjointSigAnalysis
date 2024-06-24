
"""

Please Note:
    jax.grad can call both custom_vjp & custom_jvp (whichever is defined)
    however,
    jax.vjp needs custom_vjp &
    jax.jvp needs custom_jvp
"""