""""""


from functools import reduce

def custom_bilinear(a, b, bilinear, assume_product, np_like):
    """
    Observations:
        assume_product:
            if True: (tighter bound) find min and max as a result of interval combinations
            if False:  (loser bound) extends min (left) and max (right) by finding sum of interval combinations.
    """
    a_is_interval = isinstance(a, tuple)
    b_is_interval = isinstance(b, tuple)

    if assume_product:
        # verified
        def yield_endpoints():
            a_endpoints = a if a_is_interval else (a,)
            b_endpoints = b if b_is_interval else (b,)
            for a_endpoint in a_endpoints:
                for b_endpoint in b_endpoints:
                    yield bilinear(a_endpoint, b_endpoint)
        endpoint_products = list(yield_endpoints())
        lowerBound = reduce(np_like.minimum, endpoint_products)
        upperBound = reduce(np_like.maximum, endpoint_products)
        return lowerBound, upperBound


    def positive_and_negative_parts(x):
        # note: this is element to element comparison, hence, output will be the same shape as x.
        return np_like.maximum(0, x), np_like.minimum(0, x)

    if a_is_interval and b_is_interval:
        # verified
        u, v = a
        w, x = b
        u_pos, u_neg = positive_and_negative_parts(u)
        v_pos, v_neg = positive_and_negative_parts(v)
        w_pos, w_neg = positive_and_negative_parts(w)
        x_pos, x_neg = positive_and_negative_parts(x)
        min_pairs = [(u_pos, w_pos), (v_pos, w_neg),
                     (u_neg, w_pos), (v_neg, w_neg)]
        min_vals = reduce(np_like.add, [bilinear(x, y) for x, y in min_pairs])
        max_pairs = [(v_pos, x_pos), (v_neg, w_pos),
                     (u_pos, x_neg), (u_neg, w_neg)]
        max_vals = reduce(np_like.add, [bilinear(x, y) for x, y in max_pairs])
        return min_vals, max_vals

    elif a_is_interval:
        # verified
        b_pos, b_neg = positive_and_negative_parts(b)
        min_vals = np_like.add(bilinear(a[0], b_pos),
                               bilinear(a[1], b_neg))
        max_vals = np_like.add(bilinear(a[1], b_pos),
                               bilinear(a[0], b_neg))
        return  min_vals, max_vals

    elif b_is_interval:
        # verified
        a_pos, a_neg = positive_and_negative_parts(a)
        min_vals = np_like.add(bilinear(a_pos, b[0]),
                               bilinear(a_neg, b[1]))
        max_vals = np_like.add(bilinear(a_pos, b[1]),
                               bilinear(a_neg, b[0]))
        return min_vals, max_vals

    else:
        raise NotImplementedError("Condition Not Implemented, Check arbitary_bilinear function !")
