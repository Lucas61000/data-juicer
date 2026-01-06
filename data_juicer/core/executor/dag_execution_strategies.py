import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class DAGNodeType(Enum):
    """Types of DAG nodes.

    Note: CONVERGENCE_POINT, GLOBAL_OPERATION, and REDISTRIBUTION are deprecated
    in favor of SCATTER_GATHER, which represents the complete scatter-gather pattern:
    partitions converge → global operation → redistribute.
    """

    OPERATION = "operation"
    PARTITION_OPERATION = "partition_operation"
    SCATTER_GATHER = "scatter_gather"  # NEW: replaces convergence/global/redistribution

    # Deprecated - kept for backward compatibility
    CONVERGENCE_POINT = "convergence_point"  # Deprecated: use SCATTER_GATHER
    GLOBAL_OPERATION = "global_operation"  # Deprecated: use SCATTER_GATHER
    REDISTRIBUTION = "redistribution"  # Deprecated: use SCATTER_GATHER


@dataclass
class ScatterGatherNode:
    """Represents a scatter-gather operation in partitioned execution.

    This node type encapsulates the complete scatter-gather pattern:
    1. Convergence: All partitions complete their work and converge
    2. Global Operation: A single operation runs on the gathered data
    3. Redistribution: Results are redistributed back to partitions

    This replaces the previous three separate node types (CONVERGENCE_POINT,
    GLOBAL_OPERATION, REDISTRIBUTION) which always occurred together.

    Attributes:
        operation_index: Index of the global operation in the pipeline
        operation_name: Name of the global operation
        input_partitions: List of partition IDs that converge to this operation
        output_partitions: List of partition IDs after redistribution
        node_id: Unique identifier for this scatter-gather node
    """

    operation_index: int
    operation_name: str
    input_partitions: List[int]
    output_partitions: List[int]

    @property
    def node_id(self) -> str:
        """Generate unique node ID for scatter-gather operation."""
        return f"sg_{self.operation_index:03d}_{self.operation_name}"

    def get_convergence_node_id(self) -> str:
        """Get the convergence point node ID (for backward compatibility)."""
        return f"convergence_point_{self.operation_index}"

    def get_global_op_node_id(self) -> str:
        """Get the global operation node ID (for backward compatibility)."""
        return f"op_{self.operation_index+1:03d}_{self.operation_name}_global"

    def get_redistribution_node_id(self) -> str:
        """Get the redistribution point node ID (for backward compatibility)."""
        return f"redistribution_point_{self.operation_index}"


def is_deprecated_node_type(node_type: DAGNodeType) -> bool:
    """Check if a node type is deprecated.

    Args:
        node_type: The DAGNodeType to check

    Returns:
        True if the node type is deprecated, False otherwise
    """
    return node_type in {
        DAGNodeType.CONVERGENCE_POINT,
        DAGNodeType.GLOBAL_OPERATION,
        DAGNodeType.REDISTRIBUTION,
    }


def warn_deprecated_node_type(node_type: DAGNodeType) -> None:
    """Issue a deprecation warning for deprecated node types.

    Args:
        node_type: The deprecated DAGNodeType
    """
    if is_deprecated_node_type(node_type):
        warnings.warn(
            f"DAGNodeType.{node_type.name} is deprecated and will be removed in a future version. "
            f"Use DAGNodeType.SCATTER_GATHER instead, which represents the complete "
            f"scatter-gather pattern (convergence → global operation → redistribution).",
            DeprecationWarning,
            stacklevel=3,
        )


def is_scatter_gather_component(node_type: DAGNodeType) -> bool:
    """Check if a node type is part of the scatter-gather pattern.

    Args:
        node_type: The DAGNodeType to check

    Returns:
        True if the node is SCATTER_GATHER or a deprecated component type
    """
    return node_type in {
        DAGNodeType.SCATTER_GATHER,
        DAGNodeType.CONVERGENCE_POINT,
        DAGNodeType.GLOBAL_OPERATION,
        DAGNodeType.REDISTRIBUTION,
    }


class NodeID:
    """Utility for creating and parsing standardized node IDs.

    This class provides a consistent interface for creating node IDs across
    different node types and parsing them back into their components.

    Node ID formats:
    - Operation: "op_{idx:03d}_{name}"
    - Partition Operation: "op_{idx:03d}_{name}_partition_{pid}"
    - Scatter-Gather: "sg_{idx:03d}_{name}"
    - Global Operation (legacy): "op_{idx:03d}_{name}_global"
    - Convergence Point (legacy): "convergence_point_{idx}"
    - Redistribution (legacy): "redistribution_point_{idx}"
    """

    @staticmethod
    def for_operation(op_idx: int, op_name: str) -> str:
        """Create node ID for global operation.

        Args:
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"op_{op_idx+1:03d}_{op_name}"

    @staticmethod
    def for_partition_operation(partition_id: int, op_idx: int, op_name: str) -> str:
        """Create node ID for partition operation.

        Args:
            partition_id: Partition ID
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"op_{op_idx+1:03d}_{op_name}_partition_{partition_id}"

    @staticmethod
    def for_scatter_gather(op_idx: int, op_name: str) -> str:
        """Create node ID for scatter-gather operation.

        Args:
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"sg_{op_idx:03d}_{op_name}"

    @staticmethod
    def for_global_operation(op_idx: int, op_name: str) -> str:
        """Create node ID for global operation (legacy format).

        Deprecated: Use for_scatter_gather() instead.

        Args:
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"op_{op_idx+1:03d}_{op_name}_global"

    @staticmethod
    def for_convergence_point(conv_idx: int) -> str:
        """Create node ID for convergence point (legacy format).

        Deprecated: Use for_scatter_gather() instead.

        Args:
            conv_idx: Convergence point index

        Returns:
            Standardized node ID
        """
        return f"convergence_point_{conv_idx}"

    @staticmethod
    def for_redistribution(redist_idx: int) -> str:
        """Create node ID for redistribution point (legacy format).

        Deprecated: Use for_scatter_gather() instead.

        Args:
            redist_idx: Redistribution point index

        Returns:
            Standardized node ID
        """
        return f"redistribution_point_{redist_idx}"

    @staticmethod
    def parse(node_id: str) -> Optional[Dict[str, Any]]:
        """Parse node ID into components.

        Args:
            node_id: The node ID to parse

        Returns:
            Dictionary with node type and components, or None if invalid format

        Example:
            >>> NodeID.parse("op_001_mapper_partition_0")
            {'type': DAGNodeType.PARTITION_OPERATION, 'partition_id': 0,
             'operation_index': 0, 'operation_name': 'mapper'}

            >>> NodeID.parse("sg_002_deduplicator")
            {'type': DAGNodeType.SCATTER_GATHER, 'operation_index': 2,
             'operation_name': 'deduplicator'}
        """
        # Partition operation: op_001_mapper_name_partition_0
        match = re.match(r"op_(\d+)_(.+)_partition_(\d+)", node_id)
        if match:
            return {
                "type": DAGNodeType.PARTITION_OPERATION,
                "operation_index": int(match.group(1)) - 1,  # Convert back to 0-based
                "operation_name": match.group(2),
                "partition_id": int(match.group(3)),
            }

        # Scatter-gather: sg_002_mapper_name
        match = re.match(r"sg_(\d+)_(.+)", node_id)
        if match:
            return {
                "type": DAGNodeType.SCATTER_GATHER,
                "operation_index": int(match.group(1)),
                "operation_name": match.group(2),
            }

        # Global operation (legacy): op_001_mapper_name_global
        match = re.match(r"op_(\d+)_(.+)_global", node_id)
        if match:
            return {
                "type": DAGNodeType.GLOBAL_OPERATION,
                "operation_index": int(match.group(1)) - 1,
                "operation_name": match.group(2),
            }

        # Convergence point (legacy): convergence_point_0
        match = re.match(r"convergence_point_(\d+)", node_id)
        if match:
            return {
                "type": DAGNodeType.CONVERGENCE_POINT,
                "convergence_index": int(match.group(1)),
            }

        # Redistribution (legacy): redistribution_point_0
        match = re.match(r"redistribution_point_(\d+)", node_id)
        if match:
            return {
                "type": DAGNodeType.REDISTRIBUTION,
                "redistribution_index": int(match.group(1)),
            }

        # Regular operation: op_001_mapper_name
        match = re.match(r"op_(\d+)_(.+)", node_id)
        if match:
            return {
                "type": DAGNodeType.OPERATION,
                "operation_index": int(match.group(1)) - 1,
                "operation_name": match.group(2),
            }

        # Unknown format
        return None


class DAGExecutionStrategy(ABC):
    """Abstract base class for different DAG execution strategies."""

    @abstractmethod
    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes based on execution strategy."""
        pass

    @abstractmethod
    def get_dag_node_id(self, op_name: str, op_idx: int, **kwargs) -> str:
        """Get DAG node ID for operation based on strategy."""
        pass

    @abstractmethod
    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies between nodes based on strategy."""
        pass

    @abstractmethod
    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed based on strategy."""
        pass


class NonPartitionedDAGStrategy(DAGExecutionStrategy):
    """Strategy for non-partitioned executors (default, ray)."""

    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes for non-partitioned execution."""
        nodes = {}
        for op_idx, op in enumerate(operations):
            node_id = self.get_dag_node_id(op._name, op_idx)
            nodes[node_id] = {
                "node_id": node_id,
                "operation_name": op._name,
                "execution_order": op_idx + 1,
                "node_type": DAGNodeType.OPERATION.value,
                "partition_id": None,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }
        return nodes

    def get_dag_node_id(self, op_name: str, op_idx: int, **kwargs) -> str:
        """Get DAG node ID for non-partitioned operation."""
        return f"op_{op_idx+1:03d}_{op_name}"

    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build sequential dependencies for non-partitioned execution."""
        # Simple sequential dependencies
        for i in range(1, len(operations)):
            current_node = self.get_dag_node_id(operations[i]._name, i)
            prev_node = self.get_dag_node_id(operations[i - 1]._name, i - 1)
            if current_node in nodes and prev_node in nodes:
                nodes[current_node]["dependencies"].append(prev_node)

    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed (all dependencies completed)."""
        if node_id not in nodes:
            return False
        node = nodes[node_id]
        return all(dep in completed_nodes for dep in node["dependencies"])


class PartitionedDAGStrategy(DAGExecutionStrategy):
    """Strategy for partitioned executors (ray_partitioned).

    Args:
        num_partitions: Number of partitions to use
        use_scatter_gather: If True, uses the new simplified SCATTER_GATHER node type.
                          If False, uses the legacy three-node model (convergence,
                          global operation, redistribution). Default is False for
                          backward compatibility.
    """

    def __init__(self, num_partitions: int, use_scatter_gather: bool = False):
        self.num_partitions = num_partitions
        self.use_scatter_gather = use_scatter_gather

    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes for partitioned execution.

        Args:
            operations: List of operations in the pipeline
            **kwargs: Additional arguments including 'convergence_points'

        Returns:
            Dictionary mapping node_id to node data
        """
        if self.use_scatter_gather:
            return self._generate_dag_nodes_with_scatter_gather(operations, **kwargs)
        else:
            return self._generate_dag_nodes_legacy(operations, **kwargs)

    def _generate_dag_nodes_legacy(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes using the legacy three-node model.

        This method maintains backward compatibility with the old approach of using
        three separate nodes (convergence, global operation, redistribution).

        Deprecated: This will be removed in a future version. Use the scatter-gather
        model instead.
        """
        nodes = {}
        convergence_points = kwargs.get("convergence_points", [])

        # Generate partition-specific nodes
        for partition_id in range(self.num_partitions):
            for op_idx, op in enumerate(operations):
                node_id = self.get_dag_node_id(op._name, op_idx, partition_id=partition_id)
                nodes[node_id] = {
                    "node_id": node_id,
                    "operation_name": op._name,
                    "execution_order": op_idx + 1,
                    "node_type": DAGNodeType.PARTITION_OPERATION.value,
                    "partition_id": partition_id,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                }

        # Generate convergence points
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            nodes[conv_node_id] = {
                "node_id": conv_node_id,
                "node_type": DAGNodeType.CONVERGENCE_POINT.value,
                "convergence_idx": conv_idx,
                "operation_idx": conv_point,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }

        # Generate global operation nodes
        for conv_idx, conv_point in enumerate(convergence_points):
            if conv_point < len(operations):
                op = operations[conv_point]
                global_node_id = self.get_dag_node_id(op._name, conv_point)
                nodes[global_node_id] = {
                    "node_id": global_node_id,
                    "operation_name": op._name,
                    "execution_order": conv_point + 1,
                    "node_type": DAGNodeType.GLOBAL_OPERATION.value,
                    "partition_id": None,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                }

        # Generate redistribution points
        for conv_idx, conv_point in enumerate(convergence_points):
            redist_node_id = f"redistribution_point_{conv_idx}"
            nodes[redist_node_id] = {
                "node_id": redist_node_id,
                "node_type": DAGNodeType.REDISTRIBUTION.value,
                "redistribution_idx": conv_idx,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }

        return nodes

    def _generate_dag_nodes_with_scatter_gather(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes using the new simplified scatter-gather model.

        This method uses a single SCATTER_GATHER node type that encapsulates the
        complete pattern: convergence → global operation → redistribution.
        """
        nodes = {}
        convergence_points = kwargs.get("convergence_points", [])

        # Generate partition-specific nodes
        for partition_id in range(self.num_partitions):
            for op_idx, op in enumerate(operations):
                node_id = self.get_dag_node_id(op._name, op_idx, partition_id=partition_id)
                nodes[node_id] = {
                    "node_id": node_id,
                    "operation_name": op._name,
                    "execution_order": op_idx + 1,
                    "node_type": DAGNodeType.PARTITION_OPERATION.value,
                    "partition_id": partition_id,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                }

        # Generate scatter-gather nodes for global operations
        for conv_idx, conv_point in enumerate(convergence_points):
            if conv_point < len(operations):
                op = operations[conv_point]
                sg_node = ScatterGatherNode(
                    operation_index=conv_point,
                    operation_name=op._name,
                    input_partitions=list(range(self.num_partitions)),
                    output_partitions=list(range(self.num_partitions)),
                )

                nodes[sg_node.node_id] = {
                    "node_id": sg_node.node_id,
                    "operation_name": op._name,
                    "execution_order": conv_point + 1,
                    "node_type": DAGNodeType.SCATTER_GATHER.value,
                    "operation_index": conv_point,
                    "input_partitions": sg_node.input_partitions,
                    "output_partitions": sg_node.output_partitions,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                    "scatter_gather_node": sg_node,  # Store the node object for easy access
                }

        return nodes

    def get_dag_node_id(self, op_name: str, op_idx: int, partition_id: int = None, **kwargs) -> str:
        """Get DAG node ID for partitioned operation."""
        if partition_id is not None:
            return f"op_{op_idx+1:03d}_{op_name}_partition_{partition_id}"
        else:
            # For global operations
            return f"op_{op_idx+1:03d}_{op_name}_global"

    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies for partitioned execution.

        Args:
            nodes: Dictionary of DAG nodes to build dependencies for
            operations: List of operations in the pipeline
            **kwargs: Additional arguments including 'convergence_points'
        """
        if self.use_scatter_gather:
            self._build_dependencies_with_scatter_gather(nodes, operations, **kwargs)
        else:
            self._build_dependencies_legacy(nodes, operations, **kwargs)

    def _build_dependencies_legacy(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies using the legacy three-node model.

        Deprecated: This will be removed in a future version. Use the scatter-gather
        model instead.
        """
        convergence_points = kwargs.get("convergence_points", [])

        # Build partition-specific dependencies (within each partition)
        for partition_id in range(self.num_partitions):
            for i in range(1, len(operations)):
                current_node = self.get_dag_node_id(operations[i]._name, i, partition_id=partition_id)
                prev_node = self.get_dag_node_id(operations[i - 1]._name, i - 1, partition_id=partition_id)
                if current_node in nodes and prev_node in nodes:
                    nodes[current_node]["dependencies"].append(prev_node)

        # Build convergence dependencies (all partitions converge)
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            if conv_node_id in nodes:
                for partition_id in range(self.num_partitions):
                    dep_node = self.get_dag_node_id(operations[conv_point]._name, conv_point, partition_id=partition_id)
                    if dep_node in nodes:
                        nodes[conv_node_id]["dependencies"].append(dep_node)

        # Build global operation dependencies (after convergence)
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            global_node_id = self.get_dag_node_id(operations[conv_point]._name, conv_point)
            if global_node_id in nodes and conv_node_id in nodes:
                nodes[global_node_id]["dependencies"].append(conv_node_id)

        # Build redistribution dependencies (after global operation)
        for conv_idx, conv_point in enumerate(convergence_points):
            global_node_id = self.get_dag_node_id(operations[conv_point]._name, conv_point)
            redist_node_id = f"redistribution_point_{conv_idx}"
            if redist_node_id in nodes and global_node_id in nodes:
                nodes[redist_node_id]["dependencies"].append(global_node_id)

        # Build post-redistribution dependencies (partitions resume independently)
        for conv_idx, conv_point in enumerate(convergence_points):
            redist_node_id = f"redistribution_point_{conv_idx}"
            if redist_node_id in nodes:
                for partition_id in range(self.num_partitions):
                    for i in range(conv_point + 1, len(operations)):
                        post_node = self.get_dag_node_id(operations[i]._name, i, partition_id=partition_id)
                        if post_node in nodes:
                            nodes[post_node]["dependencies"].append(redist_node_id)

    def _build_dependencies_with_scatter_gather(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies using the simplified scatter-gather model.

        This method is much cleaner than the legacy approach:
        - Partition operations depend on previous operation in same partition
        - Scatter-gather nodes depend on ALL partitions from previous op
        - Post-scatter-gather partition ops depend on the scatter-gather node
        """
        convergence_points = kwargs.get("convergence_points", [])

        # Find all scatter-gather nodes
        sg_nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node.get("node_type") == DAGNodeType.SCATTER_GATHER.value
        }

        # Build partition-specific dependencies
        for partition_id in range(self.num_partitions):
            prev_node_id = None
            for op_idx, op in enumerate(operations):
                # Skip operations that are scatter-gather points
                if op_idx in convergence_points:
                    # Find the scatter-gather node for this operation
                    sg_node_id = None
                    for nid, node in sg_nodes.items():
                        if node.get("operation_index") == op_idx:
                            sg_node_id = nid
                            break

                    if sg_node_id:
                        # Scatter-gather node depends on all partitions from previous op
                        if prev_node_id:
                            for pid in range(self.num_partitions):
                                dep_node = self.get_dag_node_id(operations[op_idx]._name, op_idx, partition_id=pid)
                                if dep_node in nodes:
                                    nodes[sg_node_id]["dependencies"].append(dep_node)

                        # Update prev_node for next iteration
                        prev_node_id = sg_node_id
                    continue

                # Regular partition operation
                node_id = self.get_dag_node_id(op._name, op_idx, partition_id=partition_id)
                if node_id in nodes:
                    # Depends on previous node in this partition (could be partition op or scatter-gather)
                    if prev_node_id:
                        nodes[node_id]["dependencies"].append(prev_node_id)
                    prev_node_id = node_id

    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed (all dependencies completed)."""
        if node_id not in nodes:
            return False
        node = nodes[node_id]
        return all(dep in completed_nodes for dep in node["dependencies"])


def is_global_operation(operation) -> bool:
    """Check if an operation is a global operation that requires convergence."""
    # Deduplicators are typically global operations
    if hasattr(operation, "_name") and "deduplicator" in operation._name:
        return True

    # Check for explicit global operation flag
    if hasattr(operation, "is_global_operation") and operation.is_global_operation:
        return True

    return False
