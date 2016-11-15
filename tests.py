import unittest
import error_functions as ef

class TestErrorFunctions(unittest.TestCase):
    def test_linex_loss_on_tuple(self):
        self.assertAlmostEqual(ef.linex_loss_on_tuple(0, 0), 0)
        self.assertAlmostEqual(ef.linex_loss_on_tuple(17, 17), 0)
        self.assertAlmostEqual(ef.linex_loss_on_tuple(1, 0), 0.004837418)
        self.assertAlmostEqual(ef.linex_loss_on_tuple(3.7, 17.2), 1.5074255306)


    def test_linex_loss(self):
        self.assertAlmostEqual(ef.linex_loss([0], [0]), 0)
        self.assertAlmostEqual(ef.linex_loss([3.7], [17.2]), 1.5074255306)
        self.assertRaises(Exception, ef.linex_loss, [0, 0], [0, 0, 10])
        self.assertAlmostEqual(ef.linex_loss([0, 0], [0, 0]), 0)
        self.assertAlmostEqual(ef.linex_loss([0, 3.7], [0, 17.2]), 1.5074255306)



if __name__ == '__main__':
    unittest.main()
