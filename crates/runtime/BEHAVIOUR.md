# cv-runtime Behaviour

How the runtime orchestrates work across devices and processes.

## Device Discovery

On first call to `scheduler()` or `registry()`:

1. **CPU backend** is always initialized (required).
2. **GPU discovery** via `cv_hal::gpu::GpuContext::global()` — registers as default GPU.
3. **MLX discovery** (Apple Silicon) via `cv_hal::mlx::MlxContext::new()`.
4. All discovered devices go into the global `DeviceRegistry` singleton.
5. When a coordinator is active, `auto_init_devices()` registers each GPU's VRAM capacity so cross-process memory budgets work from the start.

## Task Scheduling

`TaskScheduler` is the central scheduler (singleton via `scheduler()`). It manages named `ResourceGroup` instances, each bound to a device with its own thread pool.

### Group selection (`get_best_group`)

Given a backend type and workload hint:

1. Collects global load from coordinator (200ms cache).
2. Filters groups by: backend match, device health.
3. Skips low-priority groups for `Latency` hints.
4. Selects: highest priority first, then lowest total device load.

Total device load = local group load + remote process load (from coordinator).

### GPU scheduling (wait for VRAM, then CPU fallback)

When GPU VRAM is full the scheduler **waits for memory to free up** rather than
immediately falling back to CPU. GPU work stays on GPU as long as possible.

`best_gpu_with_wait(hint, timeout)` → `(group, is_gpu)`:
1. Picks the best GPU group (WebGPU > Vulkan).
2. Checks the coordinator: is this device over memory budget?
3. If over budget, **blocks** until VRAM becomes available or the timeout expires.
4. VRAM freed in time → returns GPU group with `is_gpu=true`.
5. Timeout expired → falls back to CPU group with `is_gpu=false`.

The caller gets `is_gpu` so it knows which path was taken and can adjust
(e.g. use a CPU-optimized kernel instead of the GPU shader).

The wait uses the shared-memory epoch counter: every `release_device` or `reap_dead`
bumps the epoch, so waiters wake within ~1 ms of a release instead of blind-polling.
Backoff starts at 1 ms and caps at 50 ms between checks.

Convenience function: `best_runner_gpu_wait()` → `(RuntimeRunner, bool)` wraps
this for the common case.

### GPU wait timeout (configurable at 3 levels)

| Priority | Mechanism | Example |
|---|---|---|
| 1 (highest) | Explicit per-call | `best_gpu_with_wait(hint, Some(Duration::from_secs(10)))` |
| 2 | Process-wide | `scheduler.set_gpu_wait_timeout(Duration::from_secs(60))` |
| 3 | System-wide env var | `CV_GPU_WAIT_TIMEOUT_MS=15000` |
| 4 (default) | Hardcoded | 30 seconds |

Pass `Duration::ZERO` to fail immediately (no waiting).

From Python:
```python
PyRuntime.set_gpu_wait_timeout(15000)         # 15s for this process
PyRuntime.wait_for_gpu(0, 512, timeout_ms=5000)  # explicit per-call
```

### Fallback chain (`best_gpu_or_cpu`)

For callers that *do* want CPU fallback (legacy path):
- WebGPU groups > Vulkan groups > Default CPU group

Failed devices are excluded for 60 seconds (`DEVICE_RECOVERY_TIMEOUT`).

### Error recovery (`RuntimeRunner::run_safe`)

1. Runs closure on the assigned device.
2. On failure: reports device failure to scheduler, retries on CPU if different device.
3. Failed devices are quarantined for 60s before re-try.

## Cross-Process Coordination

Activated by environment variables:

| Variable | Mode | Backend |
|---|---|---|
| `CV_RUNTIME_COORDINATOR=/path` | File-based | `FileCoordinator` |
| `CV_RUNTIME_SHM=name` | Shared memory | `ShmCoordinator` |
| Neither | Local only | No inter-process awareness |

### ShmCoordinator (v2 layout)

18KB POSIX shared memory region with lock-free CAS operations:

```
Header (64B)         magic=0x52455449, version=2, 64 slots, 8 devices
DeviceState[8]       per-device: total_mb, used_mb, owner_mask, affinity_group
ProcessSlot[64]      per-process: state machine, heartbeat, device reservations
```

#### Slot state machine

```
EMPTY(0) --CAS--> ACQUIRING(1) --> ACTIVE(2) --CAS--> RELEASING(3) --> EMPTY(0)
```

Only one thread/process wins each CAS transition.

#### Device reservation (strict, no overcommit)

```
loop:
  current = device.used_memory_mb.load()
  if current + needed > total: return Err(OutOfMemory)
  if CAS(current, current + needed): break
```

Per-slot budgets accumulate across multiple `reserve_device` calls. `release_device` frees the entire accumulated budget.

#### GPU memory estimation

`GpuContext::estimated_memory_mb()` uses `wgpu::Device::limits().max_buffer_size` as a conservative lower bound. Override with `CV_GPU_MEMORY_MB` env var for precise control.

At scheduler startup, `auto_init_devices()` registers all discovered GPU devices with the coordinator using CAS (first process wins).

#### VRAM wait mechanism

When a device is full (`used_mb >= total_mb`), callers can block via
`wait_for_device_memory(device_idx, needed_mb, timeout)`:

1. Load device's `used_mb` and `total_mb`.
2. If `total - used >= needed`: return immediately.
3. Otherwise sleep with adaptive backoff (1 ms .. 50 ms).
4. On epoch change (another process released memory): reset backoff, re-check.
5. If deadline reached: return `Err(TimedOut)`.

This means a waiter typically unblocks within a few milliseconds of another process
calling `release_device`, because the release bumps the epoch counter.

#### Heartbeat and reaping

- Background thread sends heartbeats every 1s (monotonic timestamp).
- `reap_dead()` scans all slots: if heartbeat > 5s old AND `/proc/{pid}` is gone, CAS to RELEASING, subtract device budgets, zero slot, CAS to EMPTY.
- Reaping runs on every heartbeat cycle.
- Reaping also bumps the epoch, so any waiters blocked on a dead process's VRAM get woken.

#### Affinity groups

Processes in the same group (nonzero `affinity_group`) prefer co-placement. `best_device_for_group()` counts peers per device and picks the one with the most peers (if it has enough free memory), otherwise picks the device with the most free memory globally.

## Dispatch Observability

Every GPU/CPU routing decision emits a `RuntimeEvent::Dispatch` event to the
observability layer. This makes silent CPU fallbacks visible in profiling.

Each event records:
- `operation` — what was being dispatched (e.g. "compute_normals")
- `intended_backend` — what we wanted (Gpu or Cpu)
- `actual_backend` — what actually ran
- `reason` — why (GpuAvailable, NoGpu, VramTimeout, VramFreedAfterWait, GpuError, ForcedCpu)
- `wait_ms` — how long we waited for VRAM (0 if no wait)

### For algorithm authors

Call `record_fallback()` at the GPU→CPU fallback point:

```rust
if let Ok(ComputeDevice::Gpu(gpu)) = runner.device() {
    if let Ok(result) = my_gpu_kernel(gpu, ...) { return result; }
    record_fallback("my_kernel", runner.device_id(), DispatchReason::GpuError);
} else {
    record_fallback("my_kernel", runner.device_id(), DispatchReason::NoGpu);
}
// ... CPU path ...
```

### From Python

```python
PyRuntime.get_dispatch_log(50)  # recent dispatch decisions as list of dicts
PyRuntime.fallback_count()       # how many GPU→CPU fallbacks since startup
```

## Performance Design

The runtime is the backbone — scheduling decisions must be near-zero-cost.

### Hot-path optimizations

| Concern | Solution |
|---|---|
| Backend type lookup per group | Cached at group creation (`group.backend()`) — no registry mutex |
| Global load cache clone | `Arc<HashMap>` — clone is atomic pointer bump, not heap alloc |
| Event ring buffer | `VecDeque` + `parking_lot::Mutex` — O(1) push/pop, no O(n) shift |
| Scheduler locks | All `parking_lot::Mutex` — faster than `std::sync::Mutex` for short critical sections |
| VRAM wait loop | Epoch-driven adaptive backoff (1ms..50ms) — not busy-polling |
| Heartbeat/reap | Background thread, 1s interval — zero cost on compute path |

### What's NOT on the hot path

- `init_device()` — runs once at startup
- `reserve_device()` / `release_device()` — at process registration, not per-operation
- `reap_dead()` — background thread only
- Dispatch events — O(1) ring buffer write, only fired at scheduling decision points

## Python API

Exposed via `PyRuntime`:

| Method | Description |
|---|---|
| `get_global_load()` | Aggregate device load from all processes |
| `get_num_devices()` | Count of registered devices (CPU + GPU) |
| `get_device_info()` | List of `DeviceInfo(idx, total_mb, used_mb)` |
| `reserve_device(idx, mb)` | Reserve GPU memory (strict budget) |
| `release_device(idx)` | Release reservation |
| `join_group(id)` | Join affinity group, returns `AffinityGroup` handle |
| `best_device(needed_mb)` | Find best device considering peers, returns idx or -1 |
| `wait_for_gpu(idx, mb, timeout_ms)` | Block until VRAM available or timeout |
| `set_gpu_wait_timeout(ms)` | Set process-wide wait timeout |
| `get_gpu_wait_timeout()` | Read current timeout (ms) |

## Environment Variables

| Variable | Purpose |
|---|---|
| `CV_RUNTIME_COORDINATOR` | Path for file-based coordination |
| `CV_RUNTIME_SHM` | Name for shared-memory coordination |
| `CV_GPU_MEMORY_MB` | Override GPU memory estimate (MB) |
| `CV_GPU_WAIT_TIMEOUT_MS` | System-wide GPU wait timeout (ms, default 30000) |
| `RUSTCV_GPU_MAX_BYTES` | Max single GPU allocation (checked by HAL) |

## Memory Management

`UnifiedBuffer<T>` provides CPU+GPU coherent storage with:
- Lazy device upload (only when GPU needs it)
- Version tracking for staleness detection
- Submission-indexed garbage collection

`MemoryManager` per device handles buffer pooling and deferred destruction tied to GPU submission completion.

## Thread Model

Each `ResourceGroup` owns a `rayon`-based thread pool (`Executor`). Supports:
- Core affinity pinning
- Dynamic resizing (if `allow_dynamic_scaling`)
- Work stealing (if `allow_work_stealing`)
- Backpressure via bounded task queue
