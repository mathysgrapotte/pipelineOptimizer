from environment.to_optimize_v0 import main_loop
import unittest

dummy_parameters = {
    "p1":0.12,
    "p2":42,
    "p3":8
}

class TestMainFunction(unittest.TestCase):
    def test_is_computed_equal(self):
        self.assertEqual(main_loop(dummy_parameters, 12)[0], (12**0.12+42)*2**8)

    def test_is_correct_equal(self):
        self.assertEqual(main_loop(dummy_parameters, 12)[1], (12**0.47+38)*2**12)

if __name__ == '__main__':
    unittest.main()
