# Design Doc: Parallel Partition Processing with Actor Reuse

**Author:** Data-Juicer Team
**Created:** 2026-03-09
**Updated:** 2026-03-09
**Status:** Draft
**Branch:** `feat/parallel-partition-actor-reuse`

---

## 1. Problem Statement

### Current Behavior

The `PartitionedRayExecutor` processes partitions **sequentially**, creating new GPU actors for each partition:

```
Partition 1 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
Partition 2 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
Partition 3 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
```

### Problems

1. **Repeated Model Loading**: Heavy GPU models (e.g., VideoBLIP ~20GB) are loaded N times for N partitions
2. **GPU Idle Time**: GPUs sit idle between partitions during actor teardown/creation
3. **Poor Scalability**: Processing time scales linearly with partition count due to model loading overhead

### Impact

For a typical video processing pipeline with 3 GPU operators and 10 partitions:
- Model loading time: ~60s per operator × 3 operators × 10 partitions = **30 minutes of pure overhead**
- This overhead can exceed actual processing time for smaller datasets

### Root Cause Analysis

The issue is **not** the partitioning strategy itself. Partitioning provides:
- Memory control
- Checkpoint granularity
- Resume capability

The issue is **actor lifecycle management**:
- Actors are created per `map_batches()` call
- Each partition triggers a new `map_batches()` call
- Ray garbage collects actors after each partition completes

---

## 2. Proposed Solution

### Overview

Implement **shared actor pools with detached lifecycle** that persist across partitions:

```
┌─────────────────────────────────────────────────────────────┐
│              Shared Actor Pool (Detached)                   │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                      │
│   │ A0   │ │ A1   │ │ A2   │ │ A3   │  ← Models loaded     │
│   │GPU0  │ │GPU1  │ │GPU2  │ │GPU3  │    ONCE at start     │
│   └──────┘ └──────┘ └──────┘ └──────┘                      │
└─────────────────────────────────────────────────────────────┘
       ↑          ↑          ↑          ↑
       │          │          │          │
    ┌──┴──┐    ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
    │ P0  │    │ P1  │    │ P2  │    │ P3  │
    └─────┘    └─────┘    └─────┘    └─────┘

Partitions processed sequentially, actors reused across all
```

### Key Design Principles

1. **Keep Sequential Partition Processing**: Preserves resume capability and checkpoint granularity
2. **Detached Actor Lifecycle**: Actors persist across partitions, models load once
3. **Explicit Pool Management**: Create pools at job start, cleanup at job end
4. **Compatible with Resume**: Skip completed partitions, reuse actors for remaining

---

## 3. Detailed Design

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ActorReusePartitionExecutor                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 SharedActorPoolManager                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │ Pool: NSFW  │  │Pool: Aesth. │  │Pool: Caption│             │   │
│  │  │ 8 actors    │  │ 16 actors   │  │ 8 actors    │             │   │
│  │  │ num_gpus=1  │  │ num_gpus=0.5│  │ num_gpus=1  │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                          │
│                              │ (shared across partitions)               │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Sequential Partition Processing                 │   │
│  │                                                                  │   │
│  │  for partition_id, partition in enumerate(partitions):          │   │
│  │      if is_complete(partition_id): continue  # Resume support   │   │
│  │      result = process_with_shared_actors(partition, ops)        │   │
│  │      checkpoint(partition_id, result)                           │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Execution Flow

```
Phase 1: Initialization
─────────────────────────

Job Start
    │
    ▼
┌─────────────────────┐
│ Analyze Operators   │  Identify GPU operators and their requirements
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Create Actor Pools  │  Create detached actors, load models ONCE
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Split into Partitions│
└──────────┬──────────┘


Phase 2: Sequential Processing with Actor Reuse
───────────────────────────────────────────────
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│  P0     │ │  P1     │ ...
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────────────────────────────────────┐
│        Shared Actor Pool                │
│   (models already loaded)               │
│                                         │
│   Process P0 → Checkpoint P0            │
│   Process P1 → Checkpoint P1            │
│   Process P2 → Checkpoint P2            │
│   ...                                   │
└─────────────────────────────────────────┘


Phase 3: Cleanup
────────────────
           │
           ▼
┌─────────────────────┐
│ Cleanup Actor Pools │  Kill detached actors
└──────────┬──────────┘
           │
           ▼
        Job End
```

### 3.3 Resume Flow

```
Resume from Crash (Partition 2 was in progress)
──────────────────────────────────────────────

Checkpoint State:
├── partition_0/ ✓ (complete)
├── partition_1/ ✓ (complete)
├── partition_2/ ✗ (incomplete - no _SUCCESS marker)
└── partition_3/ ✗ (not started)

Resume:
    │
    ▼
┌─────────────────────┐
│ Scan Checkpoints    │  Find: completed=[0,1], pending=[2,3]
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Create Actor Pools  │  Models load once (fresh start)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  for partition_id in [0, 1, 2, 3]:     │
│      if partition_id in [0, 1]:        │
│          skip (already complete)        │
│      else:                              │
│          process with shared actors     │
│          checkpoint                     │
└─────────────────────────────────────────┘
           │
           ▼
    Only partitions 2, 3 processed
    Models loaded only ONCE for both
```

### 3.4 Core Components

#### 3.4.1 DetachedActorPool

```python
import ray
from ray.util.actor_pool import ActorPool
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid


@dataclass
class ActorPoolConfig:
    """Configuration for an actor pool."""
    op_class: type
    op_init_args: tuple
    op_init_kwargs: dict
    num_gpus: float
    num_cpus: float
    num_actors: int
    pool_id: str


class DetachedActorPool:
    """
    A pool of detached Ray actors that persist across partitions.

    Key features:
    - Actors have 'detached' lifetime (not garbage collected)
    - Models loaded once during actor creation
    - Explicit cleanup required at job end
    """

    def __init__(self, config: ActorPoolConfig):
        self.config = config
        self.actors: List[ray.actor.ActorHandle] = []
        self.pool: Optional[ActorPool] = None
        self._created = False

    def initialize(self):
        """Create detached actors and load models."""
        if self._created:
            return

        # Create actor class with resource requirements
        actor_cls = ray.remote(
            num_gpus=self.config.num_gpus,
            num_cpus=self.config.num_cpus,
        )(self.config.op_class)

        # Create detached actors
        for i in range(self.config.num_actors):
            actor_name = f"{self.config.pool_id}_actor_{i}"

            # Check if actor already exists (from previous run)
            try:
                actor = ray.get_actor(actor_name)
                logger.info(f"Reusing existing actor: {actor_name}")
            except ValueError:
                # Create new detached actor
                actor = actor_cls.options(
                    name=actor_name,
                    lifetime="detached",
                    max_restarts=3,
                ).remote(*self.config.op_init_args, **self.config.op_init_kwargs)
                logger.info(f"Created new actor: {actor_name}")

            self.actors.append(actor)

        self.pool = ActorPool(self.actors)
        self._created = True
        logger.info(f"Actor pool initialized: {self.config.pool_id} with {len(self.actors)} actors")

    def map_batches(self, batches: List[Any], method_name: str = "process") -> List[Any]:
        """
        Process batches using the actor pool.

        Args:
            batches: List of batches to process
            method_name: Name of the actor method to call

        Returns:
            List of processed results
        """
        if not self._created:
            raise RuntimeError("Actor pool not initialized. Call initialize() first.")

        results = []

        # Submit all batches to the pool
        for batch in batches:
            self.pool.submit(
                lambda actor, b: getattr(actor, method_name).remote(b),
                batch
            )

        # Collect results in order
        while self.pool.has_next():
            result = self.pool.get_next()
            results.append(result)

        return results

    def process_dataset(self, dataset: ray.data.Dataset, batch_size: int = 1000) -> ray.data.Dataset:
        """
        Process a Ray Dataset using the actor pool.

        This method iterates through the dataset in batches,
        submits them to the actor pool, and collects results.
        """
        if not self._created:
            raise RuntimeError("Actor pool not initialized. Call initialize() first.")

        processed_batches = []

        for batch in dataset.iter_batches(batch_size=batch_size, batch_format="pyarrow"):
            # Submit to pool
            self.pool.submit(
                lambda actor, b: actor.process.remote(b),
                batch
            )

        # Collect all results
        while self.pool.has_next():
            result = self.pool.get_next()
            processed_batches.append(result)

        # Convert back to Ray Dataset
        return ray.data.from_arrow(processed_batches)

    def cleanup(self):
        """Kill all actors in the pool."""
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Failed to kill actor: {e}")

        self.actors = []
        self.pool = None
        self._created = False
        logger.info(f"Actor pool cleaned up: {self.config.pool_id}")
```

#### 3.4.2 SharedActorPoolManager

```python
class SharedActorPoolManager:
    """
    Manages shared actor pools for GPU operators across partitions.

    Responsibilities:
    - Create actor pools for each unique GPU operator configuration
    - Reuse pools across partitions
    - Handle cleanup at job completion
    """

    def __init__(self, job_id: str = None):
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.pools: Dict[str, DetachedActorPool] = {}
        self._initialized = False

    def initialize_pools(self, ops: List[OP], num_gpus_available: int = 8):
        """
        Initialize actor pools for all GPU operators.

        Args:
            ops: List of operators in the pipeline
            num_gpus_available: Total GPUs available in cluster
        """
        if self._initialized:
            logger.info("Actor pools already initialized, skipping")
            return

        for op in ops:
            if not op.use_ray_actor():
                continue  # Skip CPU operators

            pool_key = self._get_pool_key(op)

            if pool_key in self.pools:
                continue  # Pool already created for similar operator

            # Calculate number of actors based on GPU requirements
            num_actors = self._calculate_num_actors(op, num_gpus_available)

            config = ActorPoolConfig(
                op_class=op.__class__,
                op_init_args=op._init_args,
                op_init_kwargs=op._init_kwargs,
                num_gpus=op.num_gpus or 1,
                num_cpus=op.num_cpus or 1,
                num_actors=num_actors,
                pool_id=f"{self.job_id}_{op._name}",
            )

            pool = DetachedActorPool(config)
            pool.initialize()
            self.pools[pool_key] = pool

            logger.info(f"Created actor pool for {op._name}: {num_actors} actors, {op.num_gpus} GPUs each")

        self._initialized = True

    def get_pool(self, op: OP) -> Optional[DetachedActorPool]:
        """Get the actor pool for an operator."""
        pool_key = self._get_pool_key(op)
        return self.pools.get(pool_key)

    def _get_pool_key(self, op: OP) -> str:
        """
        Generate a unique key for an operator's pool.

        Operators with same class and GPU requirements can share a pool.
        """
        return f"{op.__class__.__name__}_{op.num_gpus}_{op.num_cpus}"

    def _calculate_num_actors(self, op: OP, num_gpus_available: int) -> int:
        """Calculate optimal number of actors for an operator."""
        if op.num_gpus and op.num_gpus > 0:
            # GPU operator: actors = available_gpus / gpus_per_actor
            return max(1, int(num_gpus_available / op.num_gpus))
        else:
            # CPU operator with actor mode: use num_proc
            return op.num_proc if op.num_proc > 0 else 4

    def cleanup_all(self):
        """Cleanup all actor pools."""
        for pool_key, pool in self.pools.items():
            pool.cleanup()

        self.pools = {}
        self._initialized = False
        logger.info("All actor pools cleaned up")
```

#### 3.4.3 ActorReusePartitionExecutor

```python
class ActorReusePartitionExecutor:
    """
    Partition executor with actor reuse across partitions.

    Key features:
    - Sequential partition processing (resume-friendly)
    - Shared actor pools (models load once)
    - Per-partition checkpointing
    - Resume from last incomplete partition
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_partitions = cfg.partition.get('num_of_partitions', 10)
        self.checkpoint_dir = cfg.get('checkpoint_dir', './checkpoints')
        self.actor_manager = SharedActorPoolManager(job_id=cfg.get('job_id'))

        # Detect available GPUs
        self.num_gpus = self._detect_gpus()

    def run(self, dataset: ray.data.Dataset, ops: List[OP]) -> ray.data.Dataset:
        """
        Main execution method.

        Args:
            dataset: Input Ray Dataset
            ops: List of operators to apply

        Returns:
            Processed Ray Dataset
        """
        try:
            # Phase 1: Initialize actor pools for GPU operators
            logger.info("Phase 1: Initializing actor pools...")
            self.actor_manager.initialize_pools(ops, self.num_gpus)

            # Phase 2: Split dataset into partitions
            logger.info(f"Phase 2: Splitting dataset into {self.num_partitions} partitions...")
            partitions = dataset.split(self.num_partitions)

            # Phase 3: Process partitions sequentially with actor reuse
            logger.info("Phase 3: Processing partitions...")
            processed_partitions = []

            for partition_id, partition in enumerate(partitions):
                # Check if partition already completed (resume support)
                if self._is_partition_complete(partition_id):
                    logger.info(f"Partition {partition_id}: Loading from checkpoint (already complete)")
                    result = self._load_partition_checkpoint(partition_id)
                else:
                    logger.info(f"Partition {partition_id}: Processing...")
                    result = self._process_partition(partition, ops, partition_id)

                    # Checkpoint after successful processing
                    self._save_partition_checkpoint(partition_id, result)
                    logger.info(f"Partition {partition_id}: Checkpointed")

                processed_partitions.append(result)

            # Phase 4: Union all partitions
            logger.info("Phase 4: Merging partitions...")
            final_dataset = self._union_partitions(processed_partitions)

            return final_dataset

        finally:
            # Phase 5: Cleanup actor pools
            logger.info("Phase 5: Cleaning up actor pools...")
            self.actor_manager.cleanup_all()

    def _process_partition(
        self,
        partition: ray.data.Dataset,
        ops: List[OP],
        partition_id: int
    ) -> ray.data.Dataset:
        """
        Process a single partition using shared actor pools.

        Args:
            partition: The partition dataset to process
            ops: List of operators to apply
            partition_id: ID of this partition (for logging)

        Returns:
            Processed partition dataset
        """
        result = partition

        for op_idx, op in enumerate(ops):
            logger.debug(f"Partition {partition_id}, Op {op_idx}: {op._name}")

            if op.use_ray_actor():
                # GPU operator: use shared actor pool
                pool = self.actor_manager.get_pool(op)
                if pool is None:
                    raise RuntimeError(f"No actor pool found for operator: {op._name}")

                result = pool.process_dataset(result, batch_size=op.batch_size or 1000)
            else:
                # CPU operator: use standard Ray Data processing
                result = result.map_batches(
                    op.process,
                    batch_size=op.batch_size,
                    batch_format="pyarrow",
                )

        return result

    def _is_partition_complete(self, partition_id: int) -> bool:
        """Check if a partition checkpoint exists and is complete."""
        checkpoint_path = self._get_checkpoint_path(partition_id)
        success_marker = os.path.join(checkpoint_path, "_SUCCESS")
        return os.path.exists(success_marker)

    def _save_partition_checkpoint(self, partition_id: int, dataset: ray.data.Dataset):
        """Save partition checkpoint atomically."""
        checkpoint_path = self._get_checkpoint_path(partition_id)
        temp_path = f"{checkpoint_path}.tmp"

        # Write to temp location
        dataset.write_parquet(temp_path)

        # Atomic rename
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        os.rename(temp_path, checkpoint_path)

        # Write success marker
        with open(os.path.join(checkpoint_path, "_SUCCESS"), 'w') as f:
            f.write(datetime.now().isoformat())

    def _load_partition_checkpoint(self, partition_id: int) -> ray.data.Dataset:
        """Load partition from checkpoint."""
        checkpoint_path = self._get_checkpoint_path(partition_id)
        return ray.data.read_parquet(checkpoint_path)

    def _get_checkpoint_path(self, partition_id: int) -> str:
        """Get checkpoint path for a partition."""
        return os.path.join(self.checkpoint_dir, f"partition_{partition_id:04d}")

    def _union_partitions(self, partitions: List[ray.data.Dataset]) -> ray.data.Dataset:
        """Union all partitions into a single dataset."""
        if not partitions:
            raise ValueError("No partitions to union")

        result = partitions[0]
        for p in partitions[1:]:
            result = result.union(p)

        return result

    def _detect_gpus(self) -> int:
        """Detect number of available GPUs in the cluster."""
        try:
            resources = ray.cluster_resources()
            return int(resources.get("GPU", 0))
        except Exception:
            return 8  # Default assumption
```

---

## 4. Integration with Existing PartitionedRayExecutor

### 4.1 Minimal Changes Approach

```python
# In ray_executor_partitioned.py

class PartitionedRayExecutor:
    def __init__(self, cfg):
        self.cfg = cfg
        # ... existing init ...

        # New: Actor pool manager for GPU operator reuse
        self.actor_manager = None
        self.actor_reuse_enabled = cfg.get('partition', {}).get('actor_reuse', True)

    def run(self):
        """Modified run method with actor reuse support."""
        try:
            # Initialize shared actor pools if enabled
            if self.actor_reuse_enabled and self._has_gpu_ops():
                self._initialize_shared_actors()

            # ... existing partition processing logic ...

        finally:
            # Cleanup actors
            if self.actor_manager:
                self.actor_manager.cleanup_all()

    def _initialize_shared_actors(self):
        """Initialize shared actor pools for GPU operators."""
        self.actor_manager = SharedActorPoolManager(
            job_id=self.cfg.get('job_id', str(uuid.uuid4())[:8])
        )
        self.actor_manager.initialize_pools(self.ops, self._detect_gpus())

    def _process_partition_with_actor_reuse(self, partition, ops):
        """Process partition using shared actors for GPU ops."""
        result = partition

        for op in ops:
            if op.use_ray_actor() and self.actor_manager:
                # Use shared actor pool
                pool = self.actor_manager.get_pool(op)
                result = pool.process_dataset(result)
            else:
                # Standard processing
                result = self._apply_op_standard(result, op)

        return result
```

### 4.2 Configuration Changes

```yaml
# New configuration options

partition:
  mode: 'auto'
  num_of_partitions: 10

  # Actor reuse settings (NEW)
  actor_reuse: true                    # Enable actor reuse across partitions
  actor_pool:
    max_restarts: 3                    # Actor restart limit on failure
    reuse_across_similar_ops: true     # Share pool for ops with same config
```

---

## 5. Comparison: Before vs After

### Timeline Comparison

**Before (No Actor Reuse):**
```
Time ────────────────────────────────────────────────────────────────────▶

Partition 0:  [Load Model 60s][Process 30s][GC]
Partition 1:                                   [Load Model 60s][Process 30s][GC]
Partition 2:                                                                     [Load 60s][Process 30s]

Total: 3 × (60s + 30s) = 270s
Model loads: 3
```

**After (With Actor Reuse):**
```
Time ────────────────────────────────────────────────────────────────────▶

Actor Pool:   [Load Model 60s]─────────────────────────────────────────[Cleanup]
Partition 0:                   [Process 30s]
Partition 1:                                 [Process 30s]
Partition 2:                                              [Process 30s]

Total: 60s + 3 × 30s = 150s
Model loads: 1
```

**Speedup: 1.8x** (more significant with more partitions)

### Performance Analysis

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Model Loading | N × T | 1 × T | **Nx faster** |
| GPU Idle Time | High (between partitions) | Low (continuous) | **Significant** |
| Memory Efficiency | Good (per-partition) | Good (same) | No change |
| Resume Support | Yes | Yes | No change |
| Checkpoint Granularity | Per-partition | Per-partition | No change |

Where N = number of partitions, T = model loading time

### Benchmark Projection

**Setup:**
- 8 GPUs (A100 80GB)
- 10,000 video samples
- 10 partitions
- 3 GPU operators (NSFW ~30s load, Aesthetics ~20s load, Captioning ~60s load)
- Processing: ~100s per partition

**Before:**
```
Model loading: 10 × (30 + 20 + 60) = 1100s (~18 min)
Processing: 10 × 100s = 1000s (~17 min)
Total: ~35 minutes
```

**After:**
```
Model loading: 1 × (30 + 20 + 60) = 110s (~2 min)
Processing: 10 × 100s = 1000s (~17 min)
Total: ~19 minutes
```

**Speedup: ~1.8x**

---

## 6. Checkpointing and Resume

### Checkpoint Structure

```
checkpoints/
├── job_metadata.json              # Job-level metadata
├── partition_0000/
│   ├── data.parquet
│   └── _SUCCESS                   # Completion marker
├── partition_0001/
│   ├── data.parquet
│   └── _SUCCESS
├── partition_0002/
│   └── data.parquet               # No _SUCCESS = incomplete
└── partition_0003/                # Directory doesn't exist = not started
```

### Resume Logic

```python
def find_resume_point(checkpoint_dir: str, num_partitions: int) -> List[int]:
    """
    Find which partitions need processing.

    Returns:
        List of partition IDs that need processing
    """
    pending = []

    for partition_id in range(num_partitions):
        checkpoint_path = f"{checkpoint_dir}/partition_{partition_id:04d}"
        success_marker = f"{checkpoint_path}/_SUCCESS"

        if not os.path.exists(success_marker):
            pending.append(partition_id)

    return pending
```

### Atomic Checkpoint Write

```python
def atomic_checkpoint_write(dataset, checkpoint_path):
    """
    Write checkpoint atomically to prevent corruption.

    Steps:
    1. Write to temp location
    2. Verify write succeeded
    3. Atomic rename to final location
    4. Write success marker
    """
    temp_path = f"{checkpoint_path}.tmp.{uuid.uuid4()}"

    try:
        # Step 1: Write to temp
        dataset.write_parquet(temp_path)

        # Step 2: Verify
        test_read = ray.data.read_parquet(temp_path)
        if test_read.count() != dataset.count():
            raise RuntimeError("Checkpoint verification failed")

        # Step 3: Atomic rename
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        os.rename(temp_path, checkpoint_path)

        # Step 4: Success marker
        with open(f"{checkpoint_path}/_SUCCESS", 'w') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'row_count': dataset.count(),
            }))

    except Exception as e:
        # Cleanup temp on failure
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        raise
```

---

## 7. Error Handling

### Actor Failure Recovery

```python
class DetachedActorPool:
    def _handle_actor_failure(self, actor_idx: int, error: Exception):
        """
        Handle actor failure with restart.

        Ray's 'max_restarts' handles automatic restart,
        but we may need to re-add to pool.
        """
        logger.warning(f"Actor {actor_idx} failed: {error}")

        # Check if actor was restarted by Ray
        actor = self.actors[actor_idx]
        try:
            # Ping actor to check if alive
            ray.get(actor.ping.remote(), timeout=5)
            logger.info(f"Actor {actor_idx} recovered")
        except Exception:
            # Actor dead, create replacement
            logger.info(f"Creating replacement for actor {actor_idx}")
            self.actors[actor_idx] = self._create_actor(actor_idx)

        # Rebuild pool with updated actor list
        self.pool = ActorPool(self.actors)
```

### Partition Failure Recovery

```python
def _process_partition_with_retry(self, partition, ops, partition_id, max_retries=3):
    """Process partition with retry on failure."""
    last_error = None

    for attempt in range(max_retries):
        try:
            result = self._process_partition(partition, ops, partition_id)
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Partition {partition_id} attempt {attempt + 1} failed: {e}")

            # Check if actors need recovery
            self._recover_actors_if_needed()

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(f"Partition {partition_id} failed after {max_retries} attempts: {last_error}")
```

---

## 8. Implementation Plan

### Phase 1: Core Actor Pool Infrastructure (3-4 days)

- [ ] Implement `DetachedActorPool` class
  - [ ] Actor creation with detached lifetime
  - [ ] Batch processing methods
  - [ ] Cleanup methods
  - [ ] Unit tests

- [ ] Implement `SharedActorPoolManager` class
  - [ ] Pool key generation
  - [ ] Pool initialization logic
  - [ ] Actor count calculation
  - [ ] Unit tests

### Phase 2: Integration with PartitionedRayExecutor (2-3 days)

- [ ] Modify `PartitionedRayExecutor.__init__` for actor manager
- [ ] Modify `_process_partition` to use shared actors
- [ ] Add configuration options
- [ ] Integration tests

### Phase 3: Checkpointing Enhancements (2 days)

- [ ] Implement atomic checkpoint writes
- [ ] Implement resume logic
- [ ] Add checkpoint verification
- [ ] Recovery tests

### Phase 4: Testing & Optimization (2-3 days)

- [ ] End-to-end benchmarks
- [ ] Memory profiling
- [ ] GPU utilization monitoring
- [ ] Performance comparison tests

### Phase 5: Documentation & Production Readiness (1-2 days)

- [ ] Update user documentation
- [ ] Add logging and metrics
- [ ] Configuration validation
- [ ] Code review and merge

**Total Estimated Time: 10-14 days**

---

## 9. Testing Strategy

### Unit Tests

```python
class TestDetachedActorPool:
    def test_initialize_creates_actors(self):
        """Test that initialization creates correct number of actors."""
        pass

    def test_actors_are_detached(self):
        """Test that actors have detached lifetime."""
        pass

    def test_process_batches(self):
        """Test batch processing through actor pool."""
        pass

    def test_cleanup_kills_actors(self):
        """Test that cleanup properly kills all actors."""
        pass

    def test_reuse_existing_actors(self):
        """Test that existing detached actors are reused."""
        pass


class TestSharedActorPoolManager:
    def test_initialize_pools_for_gpu_ops(self):
        """Test pool creation for GPU operators."""
        pass

    def test_skip_cpu_ops(self):
        """Test that CPU operators don't get pools."""
        pass

    def test_pool_reuse_for_similar_ops(self):
        """Test that similar ops share pools."""
        pass


class TestActorReusePartitionExecutor:
    def test_end_to_end_processing(self):
        """Test complete pipeline execution."""
        pass

    def test_resume_from_checkpoint(self):
        """Test resuming from partial completion."""
        pass

    def test_actor_reuse_across_partitions(self):
        """Verify actors are reused (models not reloaded)."""
        pass
```

### Integration Tests

```python
class TestActorReuseIntegration:
    def test_with_real_gpu_operator(self):
        """Test with actual GPU operator (e.g., VideoNSFWFilter)."""
        pass

    def test_mixed_cpu_gpu_pipeline(self):
        """Test pipeline with both CPU and GPU operators."""
        pass

    def test_checkpoint_and_resume(self):
        """Test full checkpoint/resume cycle."""
        pass

    def test_actor_failure_recovery(self):
        """Test recovery from actor failure."""
        pass
```

### Benchmark Tests

```python
class TestPerformance:
    def test_model_loading_count(self):
        """Verify model is loaded only once across partitions."""
        # Count model loading log messages
        pass

    def test_speedup_vs_baseline(self):
        """Compare performance against current implementation."""
        pass

    def test_gpu_utilization(self):
        """Measure GPU utilization during processing."""
        pass
```

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Detached actors not cleaned up | Medium | Medium | Explicit cleanup in finally block, cleanup script for orphans |
| Actor pool exhaustion | Low | High | Queue-based submission, backpressure handling |
| Memory leak in long-running actors | Low | Medium | Periodic actor restart option, memory monitoring |
| Incompatibility with Ray upgrades | Low | High | Pin Ray version, abstract actor APIs |
| Checkpoint corruption on crash | Low | High | Atomic writes, verification, temp files |

---

## 11. Future Enhancements

1. **Dynamic Actor Scaling**: Scale actor pool based on queue depth
2. **Actor Health Monitoring**: Proactive detection of unhealthy actors
3. **Cross-Job Actor Reuse**: Reuse actors across multiple jobs (with same model)
4. **GPU Memory Optimization**: Pack multiple small models on single GPU
5. **Async Checkpointing**: Checkpoint in background without blocking processing

---

## 12. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Detached Actor** | Ray actor with `lifetime="detached"` that persists until explicitly killed |
| **Actor Pool** | Collection of actors that process work items in parallel |
| **Partition** | A subset of the dataset processed as a unit |
| **Checkpoint** | Saved state of a partition for resume capability |

### B. Configuration Reference

```yaml
partition:
  # Existing options
  mode: 'auto'                         # 'auto' | 'manual'
  num_of_partitions: 10                # Number of partitions

  # New options for actor reuse
  actor_reuse: true                    # Enable actor reuse (default: true)
  actor_pool:
    max_restarts: 3                    # Max actor restarts on failure
    reuse_across_similar_ops: true     # Share pool for same-config ops
    cleanup_on_error: true             # Cleanup actors on job failure

checkpoint:
  enabled: true
  dir: './checkpoints'
  atomic_write: true                   # Use atomic checkpoint writes
  verify_on_write: true                # Verify checkpoint after write
```

### C. Monitoring and Observability

```python
# Metrics to expose
METRICS = {
    'actor_pool_size': Gauge,           # Number of actors in pool
    'actor_restarts_total': Counter,    # Total actor restarts
    'partition_processing_time': Histogram,  # Time per partition
    'model_load_time': Histogram,       # Model loading time
    'gpu_utilization': Gauge,           # GPU utilization percentage
    'checkpoint_write_time': Histogram, # Checkpoint write time
}
```

---

## References

- [Ray Actors Documentation](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray Actor Lifetimes](https://docs.ray.io/en/latest/ray-core/actors/named-actors.html#actor-lifetimes)
- [Ray Data User Guide](https://docs.ray.io/en/latest/data/data.html)
- Data-Juicer Source: `data_juicer/core/executor/ray_executor_partitioned.py`
