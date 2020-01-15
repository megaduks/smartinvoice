import unittest
import os
from utils.rename_image import main as rename_image


class MyTestCase(unittest.TestCase):

    def test_file_rename(self):
        os.chdir('utils/unit_tests')
        os.system('cp example_invoice.png testfile.png')
        input_file_name = 'testfile.png'
        output_file_name = rename_image(input_file_name)

        self.assertTrue(os.path.exists(output_file_name))
        os.remove(output_file_name)


if __name__ == '__main__':
    unittest.main()
