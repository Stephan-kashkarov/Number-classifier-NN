import unittest
import error

class Cross_entropy(unittest.TestCase):

    def test_cross_entropy_1(self):
        err = error.cross_entropy(
            [0.2698, 0.3223, 0.4078],
            [1     , 0     , 0     ],
        )
        self.assertAlmostEqual(err, 0.965)

    def test_cross_entropy_2(self):
        err = error.cross_entropy(
            [0.3138009057197197, 0.2830999994948905, 0.40309909478538986],
            [1, 0, 0],
        )
        self.assertAlmostEqual(err, 0.872)
