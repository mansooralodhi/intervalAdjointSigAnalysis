


def flatten(args):
    argsFlat = list()
    if isinstance(args, tuple | list):
        for arg in args:
            argsFlat.extend(flatten(arg))
    else:
        argsFlat.append(args)
    return argsFlat
