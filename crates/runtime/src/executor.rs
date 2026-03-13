use crate::Result;
use cv_hal::DeviceId;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const RESIZE_TIMEOUT_SECS: u64 = 30;

/// RAII guard that decrements the inflight job counter on drop.
/// Ensures the counter is properly decremented even if the job panics.
struct JobGuard(Arc<AtomicUsize>);

impl Drop for JobGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct ExecutorConfig {
    pub num_threads: usize,
    pub name: String,
    pub work_stealing: bool,
    pub core_affinity: Option<Vec<usize>>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            name: "executor".to_string(),
            work_stealing: true,
            core_affinity: None,
        }
    }
}

pub struct Executor {
    device_id: DeviceId,
    pool: RwLock<rayon::ThreadPool>,
    config: RwLock<ExecutorConfig>,
    inflight_jobs: Arc<AtomicUsize>,
    resizing: AtomicBool,
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("device_id", &self.device_id)
            .field("num_threads", &self.config.read().num_threads)
            .field("inflight_jobs", &self.inflight_jobs.load(Ordering::Relaxed))
            .finish()
    }
}

impl Executor {
    pub fn new(device_id: DeviceId, num_threads: usize, name: &str) -> Result<Self> {
        Self::with_config(
            device_id,
            ExecutorConfig {
                num_threads,
                name: name.to_string(),
                ..Default::default()
            },
        )
    }

    pub fn with_config(device_id: DeviceId, config: ExecutorConfig) -> Result<Self> {
        let pool = Self::build_pool(&config)?;

        Ok(Self {
            device_id,
            pool: RwLock::new(pool),
            config: RwLock::new(config),
            inflight_jobs: Arc::new(AtomicUsize::new(0)),
            resizing: AtomicBool::new(false),
        })
    }

    fn build_pool(config: &ExecutorConfig) -> Result<rayon::ThreadPool> {
        let name_clone = config.name.clone();
        let cores_clone = config.core_affinity.clone();

        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .thread_name(move |idx| format!("{}-{}", name_clone, idx));

        if let Some(cores) = cores_clone {
            builder = builder.start_handler(move |idx| {
                if idx < cores.len() {
                    let core_id = core_affinity::CoreId { id: cores[idx] };
                    core_affinity::set_for_current(core_id);
                }
            });
        }

        builder
            .build()
            .map_err(|e| crate::Error::RuntimeError(format!("Failed to build thread pool: {}", e)))
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn num_threads(&self) -> usize {
        self.config.read().num_threads
    }

    pub fn work_stealing_enabled(&self) -> bool {
        self.config.read().work_stealing
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);

        let pool = self.pool.read();
        pool.spawn(move || {
            let _guard = JobGuard(inflight);
            f();
        });
    }

    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);
        let _guard = JobGuard(inflight);
        let pool = self.pool.read();
        pool.install(f)
    }

    pub fn load(&self) -> usize {
        self.inflight_jobs.load(Ordering::Relaxed)
    }

    pub fn is_resizing(&self) -> bool {
        self.resizing.load(Ordering::Relaxed)
    }

    pub fn resize(&self, new_num_threads: usize) -> Result<()> {
        if self
            .resizing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(crate::Error::RuntimeError(
                "Executor is already being resized".into(),
            ));
        }

        let result = self.do_resize(new_num_threads);

        self.resizing.store(false, Ordering::Relaxed);
        result
    }

    fn do_resize(&self, new_num_threads: usize) -> Result<()> {
        let timeout = Duration::from_secs(RESIZE_TIMEOUT_SECS);
        let start = Instant::now();
        let mut sleep_duration = Duration::from_micros(100);
        let max_sleep = Duration::from_millis(100);

        while self.load() > 0 {
            if start.elapsed() > timeout {
                return Err(crate::Error::RuntimeError(
                    "Timeout waiting for jobs to complete during resize".into(),
                ));
            }
            std::thread::sleep(sleep_duration);
            sleep_duration = (sleep_duration * 2).min(max_sleep);
        }

        let mut config = self.config.write();
        config.num_threads = new_num_threads;

        let new_pool = Self::build_pool(&config)?;
        *self.pool.write() = new_pool;

        Ok(())
    }

    pub fn set_core_affinity(&self, cores: Vec<usize>) -> Result<()> {
        if self
            .resizing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(crate::Error::RuntimeError(
                "Executor is already being resized".into(),
            ));
        }

        let result = self.do_set_core_affinity(cores);

        self.resizing.store(false, Ordering::Relaxed);
        result
    }

    fn do_set_core_affinity(&self, cores: Vec<usize>) -> Result<()> {
        let timeout = Duration::from_secs(RESIZE_TIMEOUT_SECS);
        let start = Instant::now();
        let mut sleep_duration = Duration::from_micros(100);
        let max_sleep = Duration::from_millis(100);

        while self.load() > 0 {
            if start.elapsed() > timeout {
                return Err(crate::Error::RuntimeError(
                    "Timeout waiting for jobs to complete during core affinity change".into(),
                ));
            }
            std::thread::sleep(sleep_duration);
            sleep_duration = (sleep_duration * 2).min(max_sleep);
        }

        let mut config = self.config.write();
        config.core_affinity = Some(cores);

        let new_pool = Self::build_pool(&config)?;
        *self.pool.write() = new_pool;

        Ok(())
    }

    pub fn spawn_async<F, Fut>(&self, f: F) -> Result<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);

        let handle = tokio::runtime::Handle::try_current().map_err(|_| {
            crate::Error::RuntimeError("No Tokio runtime available for spawn_async".into())
        })?;

        handle.spawn(async move {
            struct JobGuard(Arc<AtomicUsize>);
            impl Drop for JobGuard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = JobGuard(inflight);
            f().await;
        });

        Ok(())
    }

    pub fn spawn_batch<T, F>(&self, items: &[T], f: F)
    where
        T: Send + Sync,
        F: Fn(&T) + Send + Sync,
    {
        let inflight = self.inflight_jobs.clone();
        inflight.fetch_add(1, Ordering::SeqCst);
        let _guard = JobGuard(inflight);

        let pool = self.pool.read();
        pool.install(|| {
            use rayon::prelude::*;
            items.par_iter().for_each(|item| {
                f(item);
            });
        });
    }
}

pub struct ExecutorPool {
    device_id: DeviceId,
    executors: Vec<Arc<Executor>>,
}

impl ExecutorPool {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            executors: Vec::new(),
        }
    }

    pub fn add_executor(&mut self, executor: Arc<Executor>) {
        assert_eq!(executor.device_id(), self.device_id);
        self.executors.push(executor);
    }

    pub fn best_executor(&self) -> Option<Arc<Executor>> {
        self.executors.iter().min_by_key(|e| e.load()).cloned()
    }

    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    pub fn total_load(&self) -> usize {
        self.executors.iter().map(|e| e.load()).sum()
    }

    pub fn least_loaded(&self) -> Option<(Arc<Executor>, usize)> {
        self.executors
            .iter()
            .map(|e| (e.clone(), e.load()))
            .min_by_key(|(_, load)| *load)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new(cv_hal::DeviceId(0), 4, "test").unwrap();
        assert_eq!(executor.num_threads(), 4);
        assert_eq!(executor.load(), 0);
    }

    #[test]
    fn test_executor_spawn() {
        let executor = Executor::new(cv_hal::DeviceId(0), 2, "test").unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        let counter_clone = counter.clone();
        executor.spawn(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_executor_pool() {
        let mut pool = ExecutorPool::new(cv_hal::DeviceId(0));

        let e1 = Arc::new(Executor::new(cv_hal::DeviceId(0), 2, "pool-1").unwrap());
        let e2 = Arc::new(Executor::new(cv_hal::DeviceId(0), 2, "pool-2").unwrap());

        pool.add_executor(e1.clone());
        pool.add_executor(e2.clone());

        assert_eq!(pool.executor_count(), 2);

        let best = pool.best_executor().unwrap();
        assert_eq!(best.load(), 0);
    }
}
