import unittest
import spacy
from ..matchers import NIP_matcher, bank_number_matcher


class MatchersTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load('pl_model')

    def test_NIP_no_dashes(self):
        doc = self.nlp('numer NIP 1234567890')
        self.assertTrue('<NIP>' in NIP_matcher(doc))

    def test_NIP_dashes_pattern_3322(self):
        doc = self.nlp('numer NIP 123-456-78-90')
        self.assertTrue('<NIP>' in NIP_matcher(doc))

    def test_NIP_dashes_pattern_3223(self):
        doc = self.nlp('numer NIP 123-45-67-890')
        self.assertTrue('<NIP>' in NIP_matcher(doc))

    def test_NIP_with_country_prefix(self):
        doc = self.nlp('numer NIP PL 1234567890')
        self.assertTrue('<NIP>' in NIP_matcher(doc))

    def test_invalid_NIP(self):
        doc = self.nlp('numer NIP 12-34-56-78-90')
        self.assertFalse('<NIP>' in NIP_matcher(doc))

    def test_no_valid_NIP(self):
        doc = self.nlp('kod pocztowy 62-090')
        self.assertFalse('<NIP>' in NIP_matcher(doc))

    def test_NIP_concatenated_with_text(self):
        doc = self.nlp('numer NIP 1234567890dodatkowytekst')
        self.assertTrue('<NIP>' in NIP_matcher(doc))

    def test_two_NIPs(self):
        doc = self.nlp('NIP sprzedawcy 123-45-67-890 i NIP klienta 0987654321')
        self.assertTrue(NIP_matcher(doc).count('<NIP>') == 2)

    def test_bank_account_spaces(self):
        doc = self.nlp('BRE BANK: 12 1234 1234 1234 1234 1234 1234')
        self.assertTrue('<BANK_NO>' in bank_number_matcher(doc))

    def test_bank_account_no_spaces(self):
        doc = self.nlp('BRE BANK: 12123412341234123412341234')
        self.assertTrue('<BANK_NO>' in bank_number_matcher(doc))

    def test_bank_account_concatenated_with_text(self):
        doc = self.nlp('BRE BANK: 12123412341234123412341234dodatkowytekst')
        self.assertTrue('<BANK_NO>' in bank_number_matcher(doc))


if __name__ == '__main__':
    unittest.main()
