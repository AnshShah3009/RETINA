pub type FnPtr = fn(&[f64]) -> Vec<f64>;

pub struct JitCompiler {
    funcs: std::collections::HashMap<String, FnPtr>,
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            funcs: std::collections::HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, func: FnPtr) {
        self.funcs.insert(name.to_string(), func);
    }

    pub fn get(&self, name: &str) -> Option<FnPtr> {
        self.funcs.get(name).copied()
    }
}

pub fn vectorize<F>(func: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    move |x: &[f64]| x.iter().map(|&v| func(v)).collect()
}

pub fn broadcast<F>(scalar: f64, func: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
{
    move |x: &[f64]| x.iter().map(|&v| func(v, scalar)).collect()
}

pub fn zip_with<F>(func: F) -> impl Fn(&[f64], &[f64]) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
{
    move |a: &[f64], b: &[f64]| a.iter().zip(b.iter()).map(|(&x, &y)| func(x, y)).collect()
}

pub fn parallel_map<F>(func: F, data: &[f64]) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    use rayon::prelude::*;
    data.par_iter().map(|&x| func(x)).collect()
}

pub fn parallel_zip_with<F>(func: F, a: &[f64], b: &[f64]) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    use rayon::prelude::*;
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| func(x, y))
        .collect()
}
