import unittest
import error

class TestError(unittest.TestCase):

    def test_cross_entropy_1(self):
        err = error.cross_entropy(
            [0.2698, 0.3223, 0.4078],
            [1     , 0     , 0     ],
        )
        self.assertAlmostEqual(err, 0.965)
