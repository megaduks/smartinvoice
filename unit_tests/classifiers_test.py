import unittest

from classifiers import *
from settings import INVOICE_IMAGE_MODEL


class ClassifiersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(ClassifiersTestCase, self).__init__(*args, **kwargs)

        self.model = InvoicePhotoClassifier(INVOICE_IMAGE_MODEL)

    def test_invoice_image_classifier(self):
        puppy_image = Path('/data/test/puppy.jpg')
        book_image = Path('/data/test/book.jpeg')
        invoice_image = Path('/data/test/invoice.jpeg')

        self.assertFalse(self.model.predict(puppy_image))
        self.assertFalse(self.model.predict(book_image))
        self.assertTrue(self.model.predict(invoice_image))


if __name__ == '__main__':
    unittest.main()
