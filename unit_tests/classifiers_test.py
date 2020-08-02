import unittest

from classifiers import InvoicePhotoClassifier, process_invoice
from settings import INVOICE_IMAGE_MODEL
from pathlib import Path


class ClassifiersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(ClassifiersTestCase, self).__init__(*args, **kwargs)

        self.model = InvoicePhotoClassifier(INVOICE_IMAGE_MODEL)

    def test_invoice_image_classifier(self):
        # unfortunately, file paths must be absolute for Ludwig classifier
        project_root = Path('/home/mikolaj/Research/smartinvoice')
        puppy_image = project_root / 'data/test/puppy.jpg'
        book_image = project_root / 'data/test/book.jpeg'
        invoice_image = project_root / 'data/test/invoice.jpeg'

        self.assertFalse(self.model.predict(puppy_image))
        self.assertFalse(self.model.predict(book_image))
        self.assertTrue(self.model.predict(invoice_image))

    def test_invoice_processing(self):
        ocr_file = Path('data/test/faktura.txt')
        result = process_invoice(ocr_file)

        expected = [
            {'numer_faktury': '01/07/2020'},
            {'typ_faktury': 'Faktura VAT'},
            {'nazwa_nabywcy': 'Baiga Mikołaj Morzy'},
            {'data_wystawienia': '10/08/2020'},
            {'miara': 'Szt'},
            {'stawka_podatku': '23%'},
            {'forma_platnosci': 'przelew'},
            {'razem_kwota_brutto': '6115,00'},
            {'numer_konta_sprzedawcy': '10 1234 5678 1234 5678 1234 5678'},
        ]

        for e in expected:
            self.assertTrue(e in result)


if __name__ == '__main__':
    unittest.main()
