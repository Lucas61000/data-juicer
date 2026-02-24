#!/usr/bin/env python3
"""
Operation Reordering Strategy for the core optimizer.

This strategy analyzes dependencies between operations and reorders them
for optimal performance, prioritizing cheap filters before expensive ones.

Key principles:
1. Mappers that modify text must run before filters that read text
2. Cheap filters (no model) should run before expensive filters (with model)
3. Early filtering reduces data volume for subsequent expensive operations
"""

from typing import Any, Dict, List

from loguru import logger

from ..pipeline_ast import OpNode, OpType, PipelineAST
from .strategy import OptimizationStrategy, register_strategy

# Filters that are cheap (no model required, fast computation)
CHEAP_FILTERS = {
    "text_length_filter",
    "words_num_filter",
    "alphanumeric_filter",
    "average_line_length_filter",
    "maximum_line_length_filter",
    "character_repetition_filter",
    "word_repetition_filter",
    "special_characters_filter",
    "suffix_filter",
    "token_num_filter",
}

# Filters that are expensive (require models or heavy computation)
EXPENSIVE_FILTERS = {
    "perplexity_filter",
    "language_id_score_filter",
    "flagged_words_filter",
    "stopwords_filter",
    "text_pair_similarity_filter",
    "phrase_grounding_recall_filter",
    "image_aesthetics_filter",
    "image_nsfw_filter",
    "image_watermark_filter",
    "video_aesthetics_filter",
    "video_nsfw_filter",
    "audio_duration_filter",
    "audio_nmf_snr_filter",
}

# Mappers that modify text content
TEXT_MODIFYING_MAPPERS = {
    "clean_html_mapper",
    "clean_links_mapper",
    "clean_email_mapper",
    "clean_copyright_mapper",
    "expand_macro_mapper",
    "fix_unicode_mapper",
    "punctuation_normalization_mapper",
    "whitespace_normalization_mapper",
    "remove_repeat_sentences_mapper",
    "remove_specific_chars_mapper",
    "remove_table_text_mapper",
    "remove_long_words_mapper",
    "remove_words_with_incorrect_substrings_mapper",
    "sentence_split_mapper",
    "chinese_convert_mapper",
    "nlpaug_en_mapper",
    "nlpcda_zh_mapper",
}


@register_strategy("op_reorder")
class OpReorderStrategy(OptimizationStrategy):
    """
    Strategy that reorders operations based on dependency analysis and cost optimization.

    Key features:
    1. Preserves mapper order (mappers often have implicit dependencies)
    2. Moves cheap filters before expensive filters
    3. Ensures filters run after mappers that modify their input
    4. Optimizes for early filtering to reduce data volume
    """

    def __init__(self):
        """Initialize the operation reordering strategy."""
        super().__init__(name="op_reorder")

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """
        Apply operation reordering to the pipeline AST.

        The reordering strategy:
        1. Keep all mappers in their original order (they may have implicit dependencies)
        2. Move cheap filters to run as early as possible (after any mappers they depend on)
        3. Keep expensive filters after cheap filters

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST with reordered operations
        """
        if not ast.root or not ast.root.children:
            return ast

        # Get all operations from the AST
        operations = self._extract_operations(ast.root)
        if len(operations) <= 1:
            return ast

        # Log original order
        original_order = [op.name for op in operations]
        logger.debug(f"Original order: {original_order}")

        # Separate mappers and filters
        mappers = [op for op in operations if op.op_type == OpType.MAPPER]
        filters = [op for op in operations if op.op_type == OpType.FILTER]

        if not filters:
            # No filters to reorder
            return ast

        # Sort filters by cost (cheap first)
        sorted_filters = sorted(filters, key=lambda op: self._get_filter_cost(op))

        # Build optimal order:
        # 1. All mappers in original order
        # 2. Cheap filters (can run after mappers)
        # 3. Expensive filters
        optimal_operations = []

        # If there are text-modifying mappers, all text filters must come after them
        # For simplicity, keep mappers first, then sorted filters
        if mappers:
            optimal_operations.extend(mappers)
        optimal_operations.extend(sorted_filters)

        optimal_order = [op.name for op in optimal_operations]

        # Check if order changed
        if original_order != optimal_order:
            # Log the reordering
            cheap = [f.name for f in sorted_filters if self._is_cheap_filter(f)]
            expensive = [f.name for f in sorted_filters if not self._is_cheap_filter(f)]
            if cheap and expensive:
                logger.info(f"Reordering: moving cheap filters {cheap} before expensive filters {expensive}")
            logger.debug(f"New order: {optimal_order}")

        # Build new AST
        return self._build_reordered_ast(ast, optimal_operations)

    def _extract_operations(self, root: OpNode) -> List[OpNode]:
        """Extract all operations from the AST in order."""
        operations = []

        def collect_ops(node: OpNode):
            if node.op_type in [OpType.MAPPER, OpType.FILTER]:
                operations.append(node)
            for child in node.children:
                collect_ops(child)

        collect_ops(root)
        return operations

    def _get_filter_cost(self, operation: OpNode) -> int:
        """
        Get cost score for a filter (lower = cheaper = higher priority).

        Cost levels:
        0-9: Cheap filters (no model, fast)
        10-19: Medium filters
        20+: Expensive filters (model required, slow)
        """
        op_name = operation.name.lower()
        config = operation.config or {}

        # Check if it requires a model
        has_model = any(
            config.get(key) is not None for key in ["model_key", "sp_model_key", "kl_model_key", "hf_model"]
        )

        # Explicitly cheap filters
        if op_name in CHEAP_FILTERS:
            return 1

        # Explicitly expensive filters
        if op_name in EXPENSIVE_FILTERS:
            return 25

        # Model-based filters are expensive
        if has_model:
            return 20

        # GPU-required filters are expensive
        if config.get("accelerator") == "cuda" or config.get("num_gpus", 0) > 0:
            return 22

        # Default: medium cost
        return 10

    def _is_cheap_filter(self, operation: OpNode) -> bool:
        """Check if a filter is cheap (no model, fast computation)."""
        return self._get_filter_cost(operation) < 10

    def _build_reordered_ast(self, original_ast: PipelineAST, operations: List[OpNode]) -> PipelineAST:
        """Build a new AST with operations in the specified order."""
        new_ast = PipelineAST()
        new_ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        current = new_ast.root
        for op in operations:
            new_op = OpNode(name=op.name, op_type=op.op_type, config=op.config.copy() if op.config else {})
            current.children = [new_op]
            new_op.parent = current
            current = new_op

        return new_ast

    def get_reorder_analysis(self, operations: List[OpNode]) -> Dict[str, Any]:
        """
        Analyze the potential benefits of reordering operations.

        Args:
            operations: List of operation nodes

        Returns:
            Dictionary with reordering analysis
        """
        filters = [op for op in operations if op.op_type == OpType.FILTER]
        mappers = [op for op in operations if op.op_type == OpType.MAPPER]

        cheap_filters = [f for f in filters if self._is_cheap_filter(f)]
        expensive_filters = [f for f in filters if not self._is_cheap_filter(f)]

        # Check if there's potential for improvement
        potential_improvement = False
        for i, op in enumerate(operations):
            if op in expensive_filters:
                # Check if any cheap filter comes after this expensive one
                for j in range(i + 1, len(operations)):
                    if operations[j] in cheap_filters:
                        potential_improvement = True
                        break

        return {
            "total_operations": len(operations),
            "mapper_count": len(mappers),
            "filter_count": len(filters),
            "cheap_filters": [f.name for f in cheap_filters],
            "expensive_filters": [f.name for f in expensive_filters],
            "potential_improvement": potential_improvement,
            "recommendation": (
                "Move cheap filters before expensive filters" if potential_improvement else "Order is already optimal"
            ),
        }
