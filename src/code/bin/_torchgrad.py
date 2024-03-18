

import torch
from torch.autograd.functional import jacobian


def f1(x, y):
    return x * y

def f2(x, y):
    out = f1(x, y)
    return torch.asarray([torch.max(out[0]), torch.max(out[1])])



# x = [5.0, 10.0]
# y = [3.0, 7.0]
# x = [[x, x],
#      [x, x]]
# x = torch.asarray(x)
# y = [[y, y],
#      [y, y]]
# y = torch.asarray(y)

print(x.shape)
output = f1(x, y)
print(output.shape)
output = f2(x, y)
print(output.shape)

print("*" * 30)
jacob = jacobian(f1, (x, y))
print(jacob[0])

print("*" * 30)
jacob = jacobian(f2, (x, y))
print(jacob[0])
