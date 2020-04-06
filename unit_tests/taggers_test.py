import unittest
import spacy

from spacy_langdetect import LanguageDetector
from taggers import InvoiceTypeTagger


class TaggersTestCase(unittest.TestCase):

    def test_language_detection(self):
        test = {
            'en': 'Invoice for ACME.com payable until the end of the month',
            'pl': 'Faktura kwota brutto termin płatności',
            'de': 'Rechnung für ACME.com zahlbar bis Ende des Monats',
        }

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

        for k, v in test.items():
            self.assertEqual(k, nlp(v)._.language['language'])
            self.assertGreater(nlp(v)._.language['score'], 0.75)

    def test_invoice_type(self):
        positive_tests = {
            'VAT': 'Faktura nr 12345',
            'VAT': 'Faktura VAT nr 123456',
            'VAT': 'Faktura NR: 12345',
            'PROFORMA': 'Faktura PROFORMA nr 12345',
            'PROFORMA': 'faktura pro-forma nr 12345',
            'PROFORMA': 'faktura Pro Forma nr 12345',
            'ZALICZKOWA': 'Faktura zaliczkowa nr 12345',
        }
        negative_tests = {
            'Termin zapłaty faktury VAT: 14 dni',
            'Faktura do rachunku nr 12345',
            'Faktura do paragonu nr 12345',
        }

        nlp = spacy.load('pl_model')
        nlp.add_pipe(InvoiceTypeTagger(nlp), name='invoice_type_tagger', last=True)

        for k, v in positive_tests.items():
            self.assertEqual(k, nlp(v)._.invoice_type)

        for doc in nlp.pipe(negative_tests):
            self.assertFalse(doc._.invoice_type)


if __name__ == '__main__':
    unittest.main()
