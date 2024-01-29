"""
'closure' / 'function-closure' / 'lexical-closure':
        an inner_function (first-class function) that remembers and has access to variables,
        in local scope in which it was created, even after the outer_function has finished
        executing. HOWEVER, we cannot access those variables.

source: https://www.youtube.com/watch?v=swU3c34d2NQ
read:   https://en.wikipedia.org/wiki/Closure_(computer_programming)
"""

##########################  Closure Example - 1 ############################

def outer_function():
    msg = "Hy"  # free variable
    # inner_function is a closure
    def inner_function():
        print(msg)
    return inner_function

func = outer_function()
func()
# print(func.msg())
print("-" * 50)

##########################  Closure Example - 2 ############################

def outer_function(message):
    msg = message
    def inner_function():
        print(msg)
    return inner_function

outer_function("Hi")()
outer_function("Hello")()
print("-" * 50)

##########################  Closure Example - 3 ############################

def outer_function(message):
    def inner_function():
        print(message)
    return inner_function

outer_function("Hi")()
outer_function("Hello")()
print("-" * 50)

##########################  Closure Example - 4  ############################

def add(a, b):
    return a + b
def mul(a, b):
    return a * b

def logger(function):
    def log_func(*args):
        print(f"{function.__name__} {args}", end=" : ")
        print(f"{function(*args)}")
    return log_func

add_logger = logger(add)
mul_logger = logger(mul)

add_logger(2, 3)
mul_logger(2, 3)

add_logger(4, 5)
mul_logger(4, 6)