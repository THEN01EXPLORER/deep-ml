import math
def rotation_layer(X, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    result = []
    for x,y in X:
        new_x = x*cos_a - y*sin_a
        new_y = y*cos_a + x*sin_a

        result.append([new_x,new_y])

    return result


    