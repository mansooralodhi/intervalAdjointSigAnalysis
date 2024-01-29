
"""
decorator:
            a function that takes another function as an argument and returns a third function
            without altering the source code of argument function.
            e.g. function_A -> decorator_function -> function_B
            the returned function is often called wrapper_function.

"""


########################## Call Wrapper Function Without Arguments ##################
#####################################################################################

def decorator_function(function):
    def wrapper_function():
        print(f"Executing {function.__name__}")
        return function()
    return wrapper_function

########################## Call Wrapper Function Without Decorator ##################

def display():
    print("Display Function !")

decorator_display = decorator_function(display)
decorator_display()
print("-" * 50)

########################## Call Wrapper Function With Decorator ##################

@decorator_function
def display():
    print("Display Function !")

display()
print("-" * 100)

########################## Call Wrapper Function With Arguments ##################
#####################################################################################

def decorator_function(function):
    def wrapper_function(*args, **kwargs):
        print(f"Executing {function.__name__}")
        return function(*args, **kwargs)
    return wrapper_function


########################## Call Wrapper Function Without Decorator ##################

def display_info(name, age):
    print(f"{name} has age {age}")

decorator_display = decorator_function(display_info)
decorator_display("Ali", 26)
print("-" * 50)

########################## Call Wrapper Function With Decorator ##################

@decorator_function
def display_info(name, age):
    print(f"{name} has age {age}")

display_info("MAnsoor ", 27)
print("-" * 50)









