import unittest
from scripts.run_openfold_nsight import counter_allowed

class CounterPolicyTests(unittest.TestCase):
    def test_root_capabilities_not_assumed(self):
        self.assertFalse(counter_allowed('1',0xa80425fb))
        self.assertTrue(counter_allowed('1',1<<21))
    def test_unrestricted_and_unknown_distinguished(self):
        self.assertTrue(counter_allowed('0',0))
        self.assertIsNone(counter_allowed(None,0))
        self.assertIsNone(counter_allowed('unexpected',1<<21))
