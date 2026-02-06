"""
Tests for DAG execution mixin.

This module tests the DAGExecutionMixin class which provides
DAG-based execution monitoring capabilities.
"""

import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class DAGExecutionMixinTest(DataJuicerTestCaseBase):
    """Tests for DAGExecutionMixin."""

    def test_mixin_import(self):
        """Test that DAGExecutionMixin can be imported."""
        from data_juicer.core.executor.dag_execution_mixin import (
            DAGExecutionMixin,
        )
        self.assertTrue(hasattr(DAGExecutionMixin, '_pre_execute_operations_with_dag_monitoring'))
        self.assertTrue(hasattr(DAGExecutionMixin, '_post_execute_operations_with_dag_monitoring'))


if __name__ == '__main__':
    unittest.main()
