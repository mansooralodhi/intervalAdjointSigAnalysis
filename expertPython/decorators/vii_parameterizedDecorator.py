"""
source: https://www.youtube.com/watch?v=KlBPCzcQNU8&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=38
"""

def argument_decorator(dec_arg):
    def decorator_function(original_function):
        def wrapper_function(*args, **kwargs):
            print(f"{dec_arg}: Executing before {original_function.__name__}")
            result = original_function(*args, **kwargs)
            print(f"{dec_arg}: Executing after {original_function.__name__}")
            return result
        return wrapper_function
    return decorator_function

@argument_decorator("LOG")
def display_info(name, age):
    print(f"{name} has age {age}")


display_info("main", 99)