import time
import logging
from functools import wraps
"""
note: wraps is itself a decorator
target: pass the actual function in next decorator in the chain of decorators.
"""


def logger(function):
    # logging.basicConfig(filename='{}.log'.format(function.__name__), level=logging.INFO)
    @wraps(function)
    def wrapper(*args, **kwargs):
        # logging.info(f"Ran with args: {args}, and kwargs: {kwargs}")
        print(f"Running logger; {function.__name__} with args: {args}")
        return function(*args, **kwargs)

    return wrapper

def timer(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time() - t1
        print(f"Running timer; {function.__name__} ran in {t2} sec")
        return result
    return wrapper

@logger
@timer
def display_info(name, age):
    time.sleep(1)
    print(f"{name} has age {age}")


display_info("lodhi", 27)
