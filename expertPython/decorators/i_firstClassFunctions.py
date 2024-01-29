
"""
Map-Function: a function whose arguments are (1) function and  (2) its parameters.
First-Class Functions: functions that are treated as variables (passed as argument or return as value).

Source: https://www.youtube.com/watch?v=kr0mpwqttM0
"""

########################  First-Class Function as Arguments    #########################

def square(x):
    return x * x

def cube(x):
    return x * x * x

def my_map(func, args_ls):
    results = list()
    for i in args_ls:
        results.append(func(i))
    return results

print(my_map(square, [1, 2, 3, 4, 5]))
print(my_map(cube, [1, 2, 3, 4, 5]))
print("-" * 50)

########################  First-Class Function as Return Value #########################

def logger(session: str):
    msg = "Performing"
    def print_loss(iteration, loss):
        # note: the 'session' value and 'msg' value  is stored in the print_loss function memory
        print(f"{msg} {session} iteration: {iteration}, loss: {loss}")
    return print_loss

log = logger("Training")
log(1, 20)
log(3, 60)
log = logger("Validation")
log(1, 10)
log(3, 5)

