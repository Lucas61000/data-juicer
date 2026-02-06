"""
Tests for config utilities.

This module tests the configuration utility functions.
"""

import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ConfigUtilsTest(DataJuicerTestCaseBase):
    """Tests for config utilities."""

    def test_config_utils_import(self):
        """Test that config_utils can be imported."""
        from data_juicer.utils import config_utils
        self.assertTrue(hasattr(config_utils, 'parse_cli_value'))


if __name__ == '__main__':
    unittest.main()
