import unittest
import os
import trident as T
import cntk as C
import numpy as np

os.environ['TRIDENT_BACKEND']='cntk'

class TestNormalizattion(unittest.TestCase):
    # def test_moment(self):
    #     input_var=C.input_variable((2,4,5),dtype=T.backend.floatx(), dynamic_axes=C.Axis.default_batch_axis())
    #     data= np.random.random((3, 2, 4, 5)).astype(T.backend.floatx())
    #     mean1, variance1=T.layers.moments(input_var,[0,1],keep_dims=True)
    #     mean2, variance2 = T.layers.moments2(input_var, [0,1],keep_dims=True)
    #     mean1=mean1.eval({input_var:data})
    #     mean2 = mean2.eval({input_var: data})
    #     print('mean1')
    #     print(mean1)
    #     print('mean2')
    #     print(mean2)
    #     err1=C.pow(mean1-mean2,2).eval().sum()
    #     print(err1)
    #     self.assertLessEqual(err1, 1e-8)
    #     variance1 = variance1.eval({input_var: data})
    #     variance2 = variance2.eval({input_var: data})
    #     print('variance1')
    #     print(variance1)
    #     print('variance2')
    #     print(variance2)
    #     err2 = C.pow(variance1 - variance2, 2).eval().sum()
    #     print(err2)
    #     self.assertLessEqual(err2, 1e-8)
    def test_moment(self):
        input_var=C.input_variable((2,4,5), dtype=T.get_backend.floatx(), dynamic_axes=C.Axis.default_batch_axis())
        data= np.random.random((3, 2, 4, 5)).astype(T.get_backend.floatx())
        mean1, variance1= trident.get_backend.layers.moments(input_var, [0, 1], keep_dims=True)
        mean2, variance2 = trident.get_backend.layers.moments2(input_var, [0, 1], keep_dims=True)
        mean1=mean1.eval({input_var:data})
        mean2 = mean2.eval({input_var: data})
        print('mean1')
        print(mean1)
        print('mean2')
        print(mean2)
        err1=C.pow(mean1-mean2,2).eval().sum()
        print(err1)
        self.assertLessEqual(err1, 1e-8)
        variance1 = variance1.eval({input_var: data})
        variance2 = variance2.eval({input_var: data})
        print('variance1')
        print(variance1)
        print('variance2')
        print(variance2)
        err2 = C.pow(variance1 - variance2, 2).eval().sum()
        print(err2)
        self.assertLessEqual(err2, 1e-8)


if __name__ == '__main__':
    unittest.main()
