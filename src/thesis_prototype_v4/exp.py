

class NodeType:
  pass


class BuildJaxpr:
  is_leaf = False
  customDtype = [NodeType]

  @classmethod
  def write(cls, x):
    print(cls.is_leaf)
    print(type(x) in cls.customDtype)

BuildJaxpr.is_leaf = True
x = NodeType()
BuildJaxpr.write(x)