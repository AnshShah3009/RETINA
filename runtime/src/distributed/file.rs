use super::LoadCoordinator;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

pub struct FileCoordinator {
    dir: PathBuf,
}

impl FileCoordinator {
    pub fn new(dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&dir);
        Self { dir }
    }

    fn get_pid_file(&self) -> PathBuf {
        self.dir.join(format!("{}.load", std::process::id()))
    }
}

impl LoadCoordinator for FileCoordinator {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()> {
        let path = self.get_pid_file();
        let mut content = String::new();

        for (dev_id, load) in device_load {
            content.push_str(&format!("{}:{}\n", dev_id.0, load));
        }

        let tmp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(content.as_bytes())?;
            file.sync_all()?;
        }
        fs::rename(tmp_path, path)?;

        Ok(())
    }

    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>> {
        let mut aggregate = HashMap::new();
        let now = SystemTime::now();
        let stale_timeout = Duration::from_secs(30);

        if !self.dir.exists() {
            return Ok(aggregate);
        }

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("load") {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if now.duration_since(modified).unwrap_or(Duration::ZERO) > stale_timeout {
                            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                if let Ok(pid) = stem.parse::<u32>() {
                                    if !is_process_alive(pid) {
                                        let _ = fs::remove_file(&path);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                if let Ok(mut file) = File::open(&path) {
                    let mut content = String::new();
                    if file.read_to_string(&mut content).is_ok() {
                        for line in content.lines() {
                            if line.is_empty() {
                                continue;
                            }
                            let parts: Vec<&str> = line.split(':').collect();
                            if parts.len() != 2 {
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    "Malformed load entry in {}: expected 'device_id:load', got '{}'",
                                    path.display(),
                                    line
                                );
                                continue;
                            }

                            match (parts[0].parse::<u32>(), parts[1].parse::<usize>()) {
                                (Ok(dev_id), Ok(load)) => {
                                    let entry = aggregate.entry(DeviceId(dev_id)).or_insert(0);
                                    *entry += load;
                                }
                                _ => {
                                    #[cfg(feature = "tracing")]
                                    tracing::warn!(
                                        "Failed to parse load entry in {}: '{}'",
                                        path.display(),
                                        line
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(aggregate)
    }

    fn cleanup(&self) {
        let _ = fs::remove_file(self.get_pid_file());
    }
}

#[cfg(target_os = "linux")]
fn is_process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

#[cfg(not(target_os = "linux"))]
fn is_process_alive(_pid: u32) -> bool {
    true
}
