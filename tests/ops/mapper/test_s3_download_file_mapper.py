"""
Tests for S3 download file mapper.

This module tests the S3DownloadFileMapper class.
Note: Full tests require S3 credentials and are skipped without them.
"""

import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class S3DownloadFileMapperTest(DataJuicerTestCaseBase):
    """Tests for S3DownloadFileMapper."""

    def test_mapper_import(self):
        """Test that S3DownloadFileMapper can be imported."""
        from data_juicer.ops.mapper.s3_download_file_mapper import (
            S3DownloadFileMapper,
        )
        self.assertTrue(hasattr(S3DownloadFileMapper, 'process_single'))


if __name__ == '__main__':
    unittest.main()
