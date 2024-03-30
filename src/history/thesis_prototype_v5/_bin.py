import numpy as np

class Node(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        newobj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)
        return newobj
#
# node = Node((2,3))
# print(node)

class MyClass:
    def __new__(cls, *args, **kwargs):
        print("__new__ called")
        print(cls.__name__)
        # instance = super().__new__(cls)
        # return instance

    def __init__(self, value):
        print("__init__ called")
        self.value = value


# Creating an instance of MyClass
obj = MyClass(42)
