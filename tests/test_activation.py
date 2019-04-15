import unittest
import activation as a


class Relu(unittest.TestCase):

    def test_relu_1(self):
        self.assertEqual(list(a.relu([0.5, -0.2, 0.7])), [0.5, 0, 0.7])

    def test_relu_2(self):
        self.assertEqual(list(a.relu([-3, -100, 0, 100, 1000, -2])), [0, 0, 0, 100, 1000, 0])

    def test_der_relu_1(self):
        self.assertEqual(list(a.der_relu([3, 2, 5, -1])), [1, 1, 1, 0])
    
    def test_der_relu_2(self):
        self.assertEqual(list(a.der_relu([-2, -5, 0, 3])), [0, 0, 0, 1])

class Sigmoid(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(
            list(a.sigmoid([1, 2, 3, 4, 5])),
            [
                0.7310585786300049,
                0.8807970779778823,
                0.9525741268224334,
                0.9820137900379085,
                0.9933071490757153
            ]
        )

    def test_der_sigmoid(self):
        self.assertEqual(
            list(a.der_sigmoid([1, 2, 3, 4, 5])),
            [
                0.36552928931500245,
                0.23688281808991007,
                0.11354961935990124,
                0.046572861464959536,
                0.017865830940122396
            ]
        )

class Softmax(unittest.TestCase):

    def test_softmax(self):
        self.assertEqual(list(a.softmax(list(range(1, 6)))),
        [
            0.011656230956039607,
            0.03168492079612427,
            0.0861285444362687,
            0.23412165725273662,
            0.6364086465588308
        ])

    def test_der_softmax(self):
        pass
