"""

{ lambda ; a:f32[784] b:i32[] c:f32[128] d:f32[784,128] e:f32[10] f:f32[128,10]. let
    g:f32[128] = dot_general[dimension_numbers=(([0], [0]), ([], []))] a d
    h:f32[128] = add g c
    i:f32[128] = custom_jvp_call[
                      call_jaxpr= { lambda ; j:f32[128]. let
                                    k:f32[128] = pjit[
                                    name=relu
                                    jaxpr= { lambda ; l:f32[128]. let
                                             m:f32[128] = max l 0.0 in (m,) }
                                  ] j
                                     in (k,) }
                      jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x00000237AE5B0940>
                      num_consts=0
                      symbolic_zeros=False
                    ] h
    n:bool[128] = gt h 0.0
    _:f32[128] = broadcast_in_dim[broadcast_dimensions=() shape=(128,)] 0.0
    o:f32[10] = dot_general[dimension_numbers=(([0], [0]), ([], []))] i f
    p:f32[10] = add o e
    q:f32[] = reduce_sum[axes=(0,)] p
    _:f32[] = div q 10.0
    r:f32[] = div 1.0 10.0
    s:f32[10] = broadcast_in_dim[broadcast_dimensions=() shape=(10,)] r
    t:f32[10,128] = dot_general[dimension_numbers=(([], []), ([], []))] s i
    u:f32[128,10] = transpose[permutation=(1, 0)] t
    v:f32[128] = dot_general[dimension_numbers=(([0], [1]), ([], []))] s f
    w:f32[128] = broadcast_in_dim[broadcast_dimensions=() shape=(128,)] 0.0
    x:f32[128] = select_n n w v
    y:f32[128,784] = dot_general[dimension_numbers=(([], []), ([], []))] x a
    z:f32[784,128] = transpose[permutation=(1, 0)] y
  in (x, z, s, u) }

"""