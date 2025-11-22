import math
from math import pi
from math import cos
from math import sin

input_angle = float(input("Enter angle in degrees: "))
angle_rad = math.radians(input_angle)
x = cos(angle_rad)
y = sin(angle_rad)
print(f"Cosine: {x}, Sine: {y}")