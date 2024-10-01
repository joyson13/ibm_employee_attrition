import unittest
from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data('../data/data.csv')
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1470)
        
if __name__ == '__main__':
    unittest.main()
    
# python -m unittest test.unit_test