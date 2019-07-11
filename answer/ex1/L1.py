# -*- coding: utf-8 -*-

import numpy as np


# returns an n-by-n identity matrix with ones on the main diagonal and zeros elsewhere.
def createEye(n):
    return np.eye(n)

if __name__ == "__main__":

    print(createEye(5))