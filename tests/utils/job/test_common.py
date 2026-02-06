"""
Tests for job common utilities.

This module tests the JobUtils class and common job management functions.
"""

import os
import shutil
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobUtilsTest(DataJuicerTestCaseBase):
    """Tests for JobUtils."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_job_common_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_job_utils_import(self):
        """Test that JobUtils can be imported."""
        from data_juicer.utils.job.common import JobUtils
        self.assertTrue(hasattr(JobUtils, 'load_job_summary'))
        self.assertTrue(hasattr(JobUtils, 'load_event_logs'))

    def test_list_running_jobs_import(self):
        """Test that list_running_jobs can be imported."""
        from data_juicer.utils.job.common import list_running_jobs
        self.assertTrue(callable(list_running_jobs))


if __name__ == '__main__':
    unittest.main()
