import time
import logging

def display_info(name, age):
    time.sleep(1)
    print(f"{name} has age {age}")

def logger(function):
    # logging.basicConfig(filename='{}.log'.format(function.__name__), level=logging.INFO)
    def wrapper(*args, **kwargs):
        # logging.info(f"Ran with args: {args}, and kwargs: {kwargs}")
        print(f"Running logger; {function.__name__} with args: {args}")
        return function(*args, **kwargs)

    return wrapper

def timer(function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time() - t1
        print(f"Running timer; {function.__name__} ran in {t2} sec")
        return result
    return wrapper

##################################### Stack Functions Manually #################################

print("Stacking Decorator Manually !")
print("-----------------------------")
print("-----------------------------")

name, age = "Ali", 26

print("display_info() --> logger --> wrapper() -> timer --> wrapper() \n" + "-" * 60)
timer(logger(display_info))(name, age)
print("-" * 30)

print("display_info() --> timer  --> wrapper() -> logger --> wrapper() \n" + "-" * 60)
logger(timer(display_info))(name, age)
print("-" * 30)


##################################### Stack Decorators #################################

print("Stacking Decorator !")
print("-------------------")
print("-------------------")

@timer
@logger # lower one gets executed first
def display_info(name, age):
    time.sleep(1)
    print(f"{name} has age {age}")

print("display_info() --> logger --> wrapper() -> timer --> wrapper() \n" + "-" * 60)
display_info("Ali", 26)
print("-" * 30)

@logger
@timer # lower one gets executed first
def display_info(name, age):
    time.sleep(1)
    print(f"{name} has age {age}")

print("display_info() --> timer  --> wrapper() -> logger --> wrapper() \n" + "-" * 60)
display_info("Ali", 26)
print("-" * 30)

############################################ END #####################################

"""
Question: 
            How do we pass the actual function in next decorator in the chain of decorators ???
Answer:     
            Consult file vi_nestedDecorators.py
"""
