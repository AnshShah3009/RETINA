use rayon::ThreadPoolBuilder;
use std::env;
use std::sync::{Mutex, OnceLock};

static THREAD_POOL_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize the global Rayon thread pool used by CPU-parallel routines.
///
/// Priority:
/// 1. `num_threads` argument
/// 2. `RUSTCV_CPU_THREADS` environment variable
/// 3. Rayon default
pub fn init_global_thread_pool(num_threads: Option<usize>) -> Result<(), String> {
    let res = THREAD_POOL_INIT.get_or_init(|| {
        let configured_threads = match num_threads {
            Some(n) => Some(n),
            None => read_cpu_threads_from_env()?,
        };

        let mut builder = ThreadPoolBuilder::new();
        if let Some(n) = configured_threads {
            if n == 0 {
                return Err("RUSTCV_CPU_THREADS must be >= 1".to_string());
            }
            builder = builder.num_threads(n);
        }

        builder
            .build_global()
            .map_err(|e| e.to_string())
            .or_else(|e| {
                if e.contains("already initialized") {
                    Ok(())
                } else {
                    Err(e)
                }
            })
    });
    res.clone()
}

pub fn current_cpu_threads() -> usize {
    rayon::current_num_threads()
}

fn read_cpu_threads_from_env() -> Result<Option<usize>, String> {
    let raw = match env::var("RUSTCV_CPU_THREADS") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(e) => return Err(format!("failed to read RUSTCV_CPU_THREADS: {e}")),
    };

    let parsed: usize = raw
        .parse()
        .map_err(|_| format!("RUSTCV_CPU_THREADS must be a positive integer, got '{raw}'"))?;
    if parsed == 0 {
        return Err("RUSTCV_CPU_THREADS must be >= 1".to_string());
    }
    Ok(Some(parsed))
}

/// A size-bucketed global buffer pool for reusing memory allocations.
pub struct BufferPool {
    // Buckets: 0: <64KB, 1: <1MB, 2: <16MB, 3: >=16MB
    // Each bucket is sharded into 8 slots to reduce Mutex contention.
    buckets: [[Mutex<Vec<Vec<u8>>>; 8]; 4],
}

static GLOBAL_BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();

impl BufferPool {
    pub fn global() -> &'static self::BufferPool {
        GLOBAL_BUFFER_POOL.get_or_init(|| BufferPool {
            buckets: std::array::from_fn(|_| std::array::from_fn(|_| Mutex::new(Vec::new()))),
        })
    }

    fn get_bucket_index(size: usize) -> usize {
        if size <= 64 * 1024 {
            0
        } else if size <= 1024 * 1024 {
            1
        } else if size <= 16 * 1024 * 1024 {
            2
        } else {
            3
        }
    }

    fn get_shard_index() -> usize {
        // Simple hash-based sharding using thread ID or similar
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        (hasher.finish() % 8) as usize
    }

    /// Get a buffer of at least `min_size` bytes.
    pub fn get(&self, min_size: usize) -> Vec<u8> {
        let bucket_idx = Self::get_bucket_index(min_size);
        let shard_idx = Self::get_shard_index();

        // Try preferred shard first, then linear search others in same bucket
        for i in 0..8 {
            let s = (shard_idx + i) % 8;
            if let Ok(mut bucket) = self.buckets[bucket_idx][s].try_lock() {
                if let Some(pos) = bucket.iter().position(|b| b.capacity() >= min_size) {
                    let mut buf = bucket.swap_remove(pos);
                    buf.clear();
                    return buf;
                }
            }
        }

        // Fallback to larger buckets if current one is empty
        for b in (bucket_idx + 1)..4 {
            for s in 0..8 {
                if let Ok(mut bucket) = self.buckets[b][s].try_lock() {
                    if let Some(pos) = bucket.iter().position(|b| b.capacity() >= min_size) {
                        let mut buf = bucket.swap_remove(pos);
                        buf.clear();
                        return buf;
                    }
                }
            }
        }

        Vec::with_capacity(min_size)
    }

    /// Return a buffer to the pool for later reuse.
    pub fn return_buffer(&self, mut buf: Vec<u8>) {
        let cap = buf.capacity();
        if cap == 0 {
            return;
        }

        let bucket_idx = Self::get_bucket_index(cap);
        let shard_idx = Self::get_shard_index();

        if let Ok(mut bucket) = self.buckets[bucket_idx][shard_idx].lock() {
            if bucket.len() < 32 {
                // Sharded limit
                buf.clear();
                bucket.push(buf);
            }
        }
    }

    /// Get a buffer wrapped in a Drop guard that returns it to the pool automatically.
    pub fn get_guarded(&self, min_size: usize) -> DropBufferPool<'_> {
        DropBufferPool::new(self, self.get(min_size))
    }
}

/// RAII guard that returns a buffer to the BufferPool when dropped.
pub struct DropBufferPool<'a> {
    pool: &'a BufferPool,
    buffer: Option<Vec<u8>>,
}

impl<'a> DropBufferPool<'a> {
    pub fn new(pool: &'a BufferPool, buffer: Vec<u8>) -> Self {
        Self {
            pool,
            buffer: Some(buffer),
        }
    }

    pub fn buffer(&self) -> &Vec<u8> {
        self.buffer.as_ref().expect(
            "DropBufferPool: buffer accessed after being taken (use-after-move logic error)",
        )
    }

    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        self.buffer.as_mut().expect(
            "DropBufferPool: buffer accessed after being taken (use-after-move logic error)",
        )
    }

    pub fn take(mut self) -> Vec<u8> {
        self.buffer
            .take()
            .expect("DropBufferPool: buffer already taken")
    }
}

impl<'a> std::ops::Deref for DropBufferPool<'a> {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

impl<'a> std::ops::DerefMut for DropBufferPool<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer_mut()
    }
}

impl<'a> Drop for DropBufferPool<'a> {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.pool.return_buffer(buf);
        }
    }
}
