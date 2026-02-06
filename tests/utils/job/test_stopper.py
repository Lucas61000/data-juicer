"""
Tests for job stopper utility.

This module tests the job stopping functionality.
"""

import os
import shutil
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JobStopperTest(DataJuicerTestCaseBase):
    """Tests for job stopper."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_job_stopper_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_stop_job_import(self):
        """Test that stop_job can be imported."""
        from data_juicer.utils.job.stopper import stop_job
        self.assertTrue(callable(stop_job))


if __name__ == '__main__':
    unittest.main()
