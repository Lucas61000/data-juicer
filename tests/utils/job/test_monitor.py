"""
Tests for job monitor utility.

This module tests the job monitoring functionality.
"""

import os
import shutil
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobMonitorTest(DataJuicerTestCaseBase):
    """Tests for job monitor."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_job_monitor_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_show_job_progress_import(self):
        """Test that show_job_progress can be imported."""
        from data_juicer.utils.job.monitor import show_job_progress
        self.assertTrue(callable(show_job_progress))


if __name__ == '__main__':
    unittest.main()
