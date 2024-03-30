
from typing import NamedTuple


class Tree:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return "Tree(x={}, y={})".format(self.x, self.y)

  def __add__(self, other):
    return Tree(self.x + other.x, self.y + other.y)

  def __sub__(self, other):
    return Tree(self.x - other.x, self.y - other.y)

  def __mul__(self, other):
    return Tree(self.x * other.x, self.y * other.y)


if __name__ == "__main__":
  import numpy as np
  print(Tree(1.0, 2.0) + Tree(1.0, 2.0))
  print(Tree(1.0, 2.0) - Tree(1.0, 2.0))
  print(Tree(1.0, 2.0) * Tree(1.0, 2.0))
  # print(np.dtype(Tree(1.0, 2.0)))