import unittest
import spacy

from matchers import NIPMatcher, BankAccountMatcher, REGONMatcher, MoneyMatcher, InvoiceNumberMatcher, DateMatcher
from tokenizers import create_custom_tokenizer


class MatchersTestCase(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super(MatchersTestCase, self).__init__(*args, **kwargs)

    def test_NIP(self):
        positive_test_strings = [
            '1234567890',
            'PL 1234567890',
            '123 456 78 90',
            '123 45 67 890',
            '123-456-78-90',
            'PL 123-456-78-90',
            '123-45-67-890',
        ]

        negative_test_strings = [
            '123 456 789',
            'AB 123 456 789',
            'abcdefghij',
            '1160 2202 0000 0001',
            '123-456-789',
            '12 34 56 78 9',
            'qwertyuiop',
            'ala ma kota',
            # 'KRS 0000046916 Sqd Rejonowy',
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = NIPMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in list(nlp.pipe(positive_test_strings)):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in list(nlp.pipe(negative_test_strings)):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_bank_account_numbers(self):
        positive_test_strings = [
            'mbank: 20123456789012345678901234',
            'mbank: PL 20123456789012345678901234',
            'mbank: 20 123456789012345678901234',
            'mbank: 20-123456789012345678901234',
            'mbank: PL 20-123456789012345678901234',
            'mbank: 123412341234123412341234',
            'mbank: 20 1234 1234 1234 1234 1234 1234',
            'mbank: 1234 1234 1234 1234 1234 1234 konto',
            # 'mbank: 20-1234-1234-1234-1234-1234-1234PL',
            # 'mbank konto:1234-1234-1234-1234-1234-1234',
        ]

        negative_test_strings = [
            'mbank: to nie jest numer konta',
            'mbank: 12345678901234567890',
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = BankAccountMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_REGON(self):
        positive_test_strings = [
            'REGON: 123456789',
            'REGON 123456789 PL',
        ]

        negative_test_strings = [
            'REGON: 123 456 789',
            'REG: 12345678',
            'REG: 123-456-789'
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = REGONMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_invoice_number(self):
        positive_test_strings = [
            'Faktura Vat nr: 1965/2019/08/0123',
            'Faktura VAT nr: 00070560/2019',
            'Faktura VAT nr 15/06/19/A ORYGINAŁ',
            'FAKTURA nr 5121/0525',
            'Faktura VAT 104/1/04/2010 oryginał',
        ]

        negative_test_strings = [
            'ala ma kota',
            'ala ma asa',
            'as ma alę'
        ]

        nlp = spacy.load('en_core_web_sm')
        matcher = InvoiceNumberMatcher(nlp)
        nlp.add_pipe(matcher, before='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_money(self):
        positive_test_strings = [
            'kwota brutto 123,45 zł',
            'kwota do zapłaty 123,90 zł',
            'pozostało do rozliczenia 239,89 PLN',
            #'kwota nierozliczona 12.34 zl'             #TODO: decimal dot is not recognized
        ]

        negative_test_strings = [
            'termin zapłaty 14 dni',
            'zakupiono 18 szt.',
            'jabłka 12 kg',
        ]

        nlp = spacy.load('pl_model')
        matcher = MoneyMatcher(nlp)
        nlp.add_pipe(matcher, after='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])

    def test_date(self):
        positive_test_strings = [
            'data sprzedaży 12-12-2019',
            'faktura z dnia 05.05.2010',
            'data płatności 01/01/2018',
            'faktura 2010-09-01',
            'data wystawienia 2010.09.01',
        ]

        negative_test_strings = [
            'Faktura FV/01/01/19',
            'Telefon 123-123-123',
            'Kontakt 11.123.2323',
        ]

        nlp = spacy.load('pl_model')
        nlp.tokenizer = create_custom_tokenizer(nlp)
        matcher = DateMatcher(nlp)
        nlp.add_pipe(matcher, after='ner')

        for doc in nlp.pipe(positive_test_strings):
            self.assertTrue(matcher.label in [e.label_ for e in doc.ents])

        for doc in nlp.pipe(negative_test_strings):
            self.assertFalse(matcher.label in [e.label_ for e in doc.ents])


if __name__ == '__main__':
    unittest.main()
