use crate::{BackendType, Capability};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPriority {
    Performance,
    Compatibility,
    Embedded,
}

pub struct DeviceManager {
    backends: Vec<BackendInfo>,
    selected_backend: Option<BackendType>,
}

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend_type: BackendType,
    pub name: String,
    pub priority: i32,
    pub available: bool,
    pub capabilities: Vec<Capability>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            selected_backend: None,
        }
    }

    pub fn detect_available(&mut self) -> &Self {
        self.backends.clear();

        if crate::cpu::CpuBackend::is_available() {
            self.backends.push(BackendInfo {
                backend_type: BackendType::Cpu,
                name: "CPU".to_string(),
                priority: 100,
                available: true,
                capabilities: vec![Capability::Compute, Capability::Simd],
            });
        }

        self
    }

    pub fn auto_select(&mut self, priority: BackendPriority) -> Result<&BackendInfo, crate::Error> {
        self.detect_available();

        if self.backends.is_empty() {
            return Err(crate::Error::backend_not_available("No backends available"));
        }

        let selected = match priority {
            BackendPriority::Performance => self
                .backends
                .iter()
                .filter(|b| b.available)
                .max_by_key(|b| b.priority),
            BackendPriority::Compatibility | BackendPriority::Embedded => self
                .backends
                .iter()
                .find(|b| b.backend_type == BackendType::Cpu),
        };

        match selected {
            Some(b) => {
                self.selected_backend = Some(b.backend_type);
                Ok(b)
            }
            None => Err(crate::Error::backend_not_available(
                "No suitable backend found",
            )),
        }
    }

    pub fn available_backends(&self) -> &[BackendInfo] {
        &self.backends
    }

    pub fn selected(&self) -> Option<BackendType> {
        self.selected_backend
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn detect_hardware() -> DeviceManager {
    let mut manager = DeviceManager::new();
    manager.detect_available();
    manager
}

pub fn get_best_backend() -> BackendType {
    let manager = detect_hardware();

    manager
        .available_backends()
        .iter()
        .filter(|b| b.available)
        .max_by_key(|b| b.priority)
        .map(|b| b.backend_type)
        .unwrap_or(BackendType::Cpu)
}
