import numpy as np

class Model:
    def __init__(self, w1, w2, b1, b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2


# all possible combinations and correct outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

w1, w2 = np.random.randn(2, 2)
b1, b2 = np.random.randn(2)