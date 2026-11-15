import unittest

from datamate.normalize import as_number, normalize, quantity_separators


class QuantityFormats(unittest.TestCase):
    def test_grouping_evidence(self):
        for examples, token in [(["2,001,500", "3,400.25"], "1,234"),
                                (["2.001.500", "3.400,25"], "1.234")]:
            self.assertEqual(as_number(token + " m3", "quantity", quantity_separators(examples)), 1234)

    def test_decimal_evidence(self):
        for examples, token in [(["1.25", "0.125"], "1.234"),
                                (["1,25", "0,125"], "1,234")]:
            self.assertEqual(as_number(token, "quantity", quantity_separators(examples)), 1.234)

    def test_conflict_or_no_evidence_is_missing(self):
        for examples in (["1.234"], ["2.001.500", "0.125"]):
            self.assertEqual(normalize("1.234", "quantity", None, None,
                                       quantity_separators(examples)), (None, True))
        self.assertEqual(normalize("0.125", "quantity", None, None), (0.125, False))
        self.assertEqual(normalize(1.234, "quantity", None, None), (1.234, False))

    def test_mixed_grouping_and_decimal(self):
        self.assertEqual(as_number("12,345.125 hộp", "quantity"), 12345.125)
        self.assertEqual(as_number("12.345,125 hộp", "quantity"), 12345.125)
