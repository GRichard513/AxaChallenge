import pandas as pd
import math

def linex_loss_on_tuple(prediction_y, real_y):
    alpha = -0.1
    error = math.exp(alpha*(y-prediction_y)) - alpha*(y-prediction_y) - 1
    return error

def linex_loss(prediction, real):
    if (len(prediction)!=len(real)):
        raise NameError("Incompatible lengths of vector for LinEx.")
    n = len(real)
    error = 0
    for i in range(0,n):
        prediction_y = prediction[i]
        real_y = real[i]
        error_y = linex_loss_on_tuple(prediction_y, real_y)
        error += error_y
    print("LinEx error : ", error)
    return error
