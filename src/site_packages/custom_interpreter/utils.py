


def flatten(args):
    argsFlat = list()
    # todo: remove tuple from the below isinstance
    #  because we don't want to flatten interval
    if isinstance(args, tuple | list):
        for arg in args:
            argsFlat.extend(flatten(arg))
    else:
        argsFlat.append(args)
    return argsFlat
