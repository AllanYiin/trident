import unittest
import os
os.environ['TRIDENT_BACKEND']='cntk'
from trident import get_backend as T
import cntk as C

class TestNormalizattion(unittest.TestCase):
    def test_get_normalization1(self):
        fn1=T.get_normalization('GroupNormalization')
        for att in dir(fn1):
            print(att, getattr(fn1, att))

        print('fn1: {0}'.format(fn1.__name__))
        self.assertEqual(fn1.__name__, 'GroupNormalization')
    def test_get_normalization2(self):
        fn2=T.get_normalization('Instance')
        print(fn2.__name__)
        self.assertEqual(fn2.__name__, 'InstanceNormalization')

    def test_get_normalization3(self):
        fn3 = T.get_normalization('b')
        print(fn3.__name__)
        self.assertEqual(fn3.__name__, 'BatchNormalization')


if __name__ == '__main__':
    unittest.main()
