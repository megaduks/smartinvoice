import unittest
import spacy

from spacy_langdetect import LanguageDetector


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


if __name__ == '__main__':
    unittest.main()
