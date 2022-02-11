import math as m
import numpy as np


class _RotationMatrix:
    """
    _RotationMatrix namespace
    """

    @staticmethod
    def matrix(theta):
        return (_RotationMatrix.rx(theta[0]) *
                _RotationMatrix.ry(theta[1]) *
                _RotationMatrix.rz(theta[2]))

    @staticmethod
    def rotate(vertices,theta):
        R = _RotationMatrix.matrix(theta)
        return R @ vertices

    @staticmethod
    def rx(theta):
        return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])
    @staticmethod
    def ry(theta):
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])
    @staticmethod
    def rz(theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])
