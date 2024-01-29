
class decorator_class(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        print(f"Executing {self.function.__name__}")
        return self.function(*args, **kwargs)


########################## Call Wrapper Class Without Decorator ##################

def display_info(name, age):
    print(f"{name} has age {age}")

decorator_display = decorator_class(display_info)
decorator_display("Ali", 26)
print("-" * 50)

########################## Call Wrapper Class With Decorator ##################

@decorator_class
def display_info(name, age):
    print(f"{name} has age {age}")

display_info("MAnsoor ", 27)
print("-" * 50)
