from collections import namedtuple

"""
namedtuple: 
            a tuple which is more readable.

namedtuple vs dict:
    difference:
        1. namedtuple is immutable while dict is mutable.
        2. namedtuple object needs all values while dict don't need all key/values.
        3. more typing in case of dict.
    similarity:
        1. assign name to each value or attribute.
        2. can call an attribute with name.
        3. can assign values by kwargs.

source: https://www.youtube.com/watch?v=GfxJYp9_nJA&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=39
"""

RGB = namedtuple(typename="RGBcolor", field_names=["red", "green", "blue"])
color1 = RGB(155, 255, 160)
color3 = RGB(blue=155, green=255, red=100)

print(RGB.__name__)
print(color1)
print(color3)
print(color3.green)

HSV = namedtuple(typename="HSVcolor", field_names=["hue", "saturation", "value"])
color1 = HSV(155, 255, 160)
color3 = HSV(saturation=155, value=255, hue=100)
print(HSV.__name__)
print(color1)
print(color3)
print(color3.saturation)
