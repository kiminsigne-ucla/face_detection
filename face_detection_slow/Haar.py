# This class defines Haar features used for face detection. There are four different types of Haar features, each
# evaluated differently, defined as sub-classes of Haar. The evaluate functions for each class are defined in
# haar_funcs.pyx and are Cythonized for faster evaluation. The appropriate evalute function is bound to each class
# dynamically.

import types
import haar_funcs


class Haar:

    def __init__(self, height, width, class_type, inverse, x, y):
        self.height = height
        self.width = width
        self.class_type = class_type
        self.inverse = inverse
        self.x = x
        self.y = y
        self.threshold = None
        self.parity = None
        self.weight = None

    def classify(self, integral_img, score=None):

        if not score:
            score = self.evaluate(integral_img)
        if self.parity * score < self.parity * self.threshold:
            image_class = 1
        else:
            image_class = -1

        return image_class


class HaarClassI(Haar):

    # Type 1: two vertical rectangles side by side. Default arrangement (inverse = 0) assigns the white square to the
    # left and the dark square to the right. Inverse = 1 sets the dark square to the left and the white square to the
    # right

    def __init__(self, height, width, class_type, inverse, x, y):
        Haar.__init__(self, height, width, class_type, inverse, x, y)


HaarClassI.evaluate = types.MethodType(haar_funcs.evaluate_classI, None, HaarClassI)


class HaarClassII(Haar):

    # Type II: Horizontal rectangles on top of each other. Default arrangement (inverse = 0) assigns dark square to top
    # and white square to bottom. Inverse = 1 assigns white square to top and dark square to bottom.

    def __init__(self, height, width, class_type, inverse, x, y):
        Haar.__init__(self, height, width, class_type, inverse, x, y)


HaarClassII.evaluate = types.MethodType(haar_funcs.evaluate_classII, None, HaarClassII)


class HaarClassIII(Haar):

    # Type 3: Three vertical rectangles side by side
    # Default arrangement (inverse = 0) assigns white-dark-white. Inverse = 1
    # assigns dark-white-dark.

    def __init__(self, height, width, class_type, inverse, x, y):
        Haar.__init__(self, height, width, class_type, inverse, x, y)


HaarClassIII.evaluate = types.MethodType(haar_funcs.evaluate_classIII, None, HaarClassIII)


class HaarClassIV(Haar):

    # Type IV: 2x2 grid of rectangles. Default arrangement (inverse = 0) assigns white squares along left-right
    # diagonal, dark squares on right-left diagonal. Inverse = 1 assigns dark squares along left-right diagonal, white
    # squares on right-left diagonal.

    def __init__(self, height, width, class_type, inverse, x, y):
        Haar.__init__(self, height, width, class_type, inverse, x, y)


HaarClassIV.evaluate = types.MethodType(haar_funcs.evaluate_classIV, None, HaarClassIV)
