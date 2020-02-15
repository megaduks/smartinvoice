import unittest
import spacy
from ..taggers import nip_tagger, bank_number_tagger, date_tagger, regon_tagger


class MatchersTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load('pl_model')

    def test_NIP_no_dashes(self):
        doc = self.nlp('numer NIP 1234567890')
        self.assertTrue('<NIP>' in nip_tagger(doc).text)

    def test_NIP_dashes_pattern_3322(self):
        doc = self.nlp('numer NIP 123-456-78-90')
        self.assertTrue('<NIP>' in nip_tagger(doc).text)

    def test_NIP_dashes_pattern_3223(self):
        doc = self.nlp('numer NIP 123-45-67-890')
        self.assertTrue('<NIP>' in nip_tagger(doc).text)

    def test_NIP_with_country_prefix(self):
        doc = self.nlp('numer NIP PL 1234567890')
        self.assertTrue('<NIP>' in nip_tagger(doc).text)

    def test_invalid_NIP(self):
        doc = self.nlp('numer NIP 12-34-56-78-90')
        self.assertFalse('<NIP>' in nip_tagger(doc).text)

    def test_no_valid_NIP(self):
        doc = self.nlp('kod pocztowy 62-090')
        self.assertFalse('<NIP>' in nip_tagger(doc).text)

    def test_NIP_concatenated_with_text(self):
        doc = self.nlp('numer NIP 1234567890dodatkowytekst')
        self.assertTrue('<NIP>' in nip_tagger(doc).text)

    def test_two_NIPs(self):
        doc = self.nlp('NIP sprzedawcy 123-45-67-890 i NIP klienta 0987654321')
        self.assertTrue(nip_tagger(doc).text.count('<NIP>') == 2)

    def test_bank_account_spaces(self):
        doc = self.nlp('BRE BANK: 12 1234 1234 1234 1234 1234 1234')
        self.assertTrue('<BANK_NO>' in bank_number_tagger(doc).text)

    def test_bank_account_no_spaces(self):
        doc = self.nlp('BRE BANK: 12123412341234123412341234')
        self.assertTrue('<BANK_NO>' in bank_number_tagger(doc).text)

    def test_bank_account_concatenated_with_text(self):
        doc = self.nlp('BRE BANK: 12123412341234123412341234dodatkowytekst')
        self.assertTrue('<BANK_NO>' in bank_number_tagger(doc).text)

    def test_simple_date_with_dashes(self):
        doc = self.nlp('data sprzedaży 12-10-2017')
        self.assertTrue('<DATE>' in date_tagger(doc).text)

    def test_simple_date_with_dots(self):
        doc = self.nlp('data sprzedaży 1.1.2020')
        self.assertTrue('<DATE>' in date_tagger(doc).text)

    def test_simple_date_with_spaces(self):
        doc = self.nlp('data sprzedaży 31 12 2019')
        self.assertTrue('<DATE>' in date_tagger(doc).text)

    def test_simple_date_reversed(self):
        doc = self.nlp('data sprzedaży 2020 1 1')
        self.assertTrue('<DATE>' in date_tagger(doc).text)

    def test_simple_REGON(self):
        doc = self.nlp('Fiat Polska NIP: 123-456-77-89 REGON: 123456789')
        self.assertTrue('<REGON>' in regon_tagger(doc).text)


if __name__ == '__main__':
    unittest.main()
