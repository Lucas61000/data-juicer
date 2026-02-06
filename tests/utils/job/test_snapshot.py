"""
Tests for job snapshot utility.

This module tests the processing snapshot functionality for job status analysis.
"""

import os
import shutil
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobSnapshotTest(DataJuicerTestCaseBase):
    """Tests for job snapshot."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_job_snapshot_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_snapshot_module_import(self):
        """Test that snapshot module can be imported."""
        from data_juicer.utils.job import snapshot
        self.assertTrue(hasattr(snapshot, 'ProcessingSnapshot'))


if __name__ == '__main__':
    unittest.main()
