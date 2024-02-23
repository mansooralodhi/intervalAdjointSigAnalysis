"""

Github Source:
-------------
    1.  https://github.com/google/jax/discussions/10663
    2.  https://github.com/google/jax/blob/1b0be5095a62064820301d1fa25f3c38596e1ae2/jax/experimental/sparse/transform.py#L173
    3.  https://github.com/google/jax/discussions/9779
    4.  https://github.com/google/tree-math/issues/6

Official Documentation:
----------------------
    1.  https://estimagic.readthedocs.io/en/stable/development/eep-01-pytrees.html#difference-between-pytrees-in-jax-and-estimagic
    2.  https://jax.readthedocs.io/en/latest/autodidax.html#jax-core-machinery


Observations / Questions / Tasks:
--------------------------------
1.  We can execute a function with interval input without doing anything at all.
    Simply overwrite some basic operations and for higher order operations
    use numpy wrapper.
2.  The key challenge is to find the forward computational graph or jax.vjp
    on input type intervals.
3.  Jax perform trace while building the computational graph and use the
    input data.
4.  We need to stop looking at interval as a datatype, rather we should
    look at interval object as leaf of pytree. Hence, the first objective
    would be to create a pytree then add the functionality to flatten
    and unflatten the pytree. Note, that even a matrix is broken down
    into rows and individual elements as leaves. But how are we
    going to create pytree out of a numpy matrix ???
5.  For current scenario we consider leaf as interval or a numpy array.
    The problem is: numpy of float can be traced during construction of
    computational graph but a numpy of intervals (dtype=object) can't
    be traced.
        Question raises how to make this numpy (with interval)
        a valid jax datatype ???
        Should we fix this issue while creating Abstract Value ???
5.  Anyhow, once we have abstract values against the concrete input values
    we can use them with flattened function to find the computational
    graph or vjp.
6.  Now lefts break down the tasks:
    1.
        i.  Create a pytree with interval as leaves.
        ii. Write a flatten method that flatten everything except
            a numpy array and an interval object.
        iii.Write an unflatten function to restore the pytree.
        iv. Test the code by flattening and un-flattening nested
            structures.
    2.
        i.  Now we are going to map the input tree to abstract values.
        ii. Rather than concrete shape we need abstract shape to
            make jaxpr.
    3.
        i.  Once we have the flatten function and abstract values
            we apply the pe.trace_to_jaxpr_dynamic to get the jaxpr.



"""