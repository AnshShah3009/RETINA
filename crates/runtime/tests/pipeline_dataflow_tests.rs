//! Data actually moves through a `Pipeline` (GAPS g-i1).
//!
//! Before the fix a CPU-only pipeline could not run two chained ops at all
//! ("TransientBufferPool only supports GPU"), and on a GPU a kernel's output
//! reached a following CPU op as an empty slice while a CPU op's output was
//! never uploaded to the GPU buffer a following kernel bound.

use cv_runtime::orchestrator::RuntimeRunner;
use cv_runtime::pipeline::{BufferId, Pipeline};
use std::sync::{Arc, Mutex};

fn cpu_runner() -> RuntimeRunner {
    RuntimeRunner::Sync(cv_runtime::registry().unwrap().default_cpu().id())
}

fn gpu_runner() -> Option<RuntimeRunner> {
    let gpu = cv_runtime::registry().ok()?.default_gpu()?;
    Some(RuntimeRunner::Sync(gpu.id()))
}

type Captured = Arc<Mutex<Option<Vec<Vec<u8>>>>>;

/// A three-op CPU chain: produce A, B = 2*A, C = A + B, then capture C.
fn cpu_chain(captured: Captured) -> Pipeline {
    let (a, b, c) = (BufferId(1), BufferId(2), BufferId(3));
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a], |_| vec![vec![1, 2, 3, 4]])
        .add_cpu_op(&[a], &[b], |ins| {
            vec![ins[0].iter().map(|x| x * 2).collect()]
        })
        .add_cpu_op(&[a, b], &[c], |ins| {
            vec![ins[0].iter().zip(ins[1]).map(|(x, y)| x + y).collect()]
        })
        .add_cpu_op(&[c], &[], move |ins| {
            *captured.lock().unwrap() = Some(ins.iter().map(|s| s.to_vec()).collect());
            vec![]
        });
    for id in [a, b, c] {
        p.register_buffer(id, 4);
    }
    p.build().unwrap()
}

#[test]
fn cpu_only_chain_produces_right_numbers() {
    let captured: Captured = Arc::default();
    cpu_chain(captured.clone())
        .execute(&cpu_runner())
        .expect("CPU-only pipeline must run");
    assert_eq!(
        captured.lock().unwrap().clone(),
        Some(vec![vec![3, 6, 9, 12]])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cpu_only_chain_produces_right_numbers_async() {
    let captured: Captured = Arc::default();
    let mut handle = cpu_chain(captured.clone()).execute_async(&cpu_runner());
    handle
        .wait_for_completion()
        .await
        .expect("CPU-only async pipeline must run");
    assert_eq!(
        captured.lock().unwrap().clone(),
        Some(vec![vec![3, 6, 9, 12]])
    );
}

#[test]
fn cpu_op_inputs_keep_their_order() {
    let (a, b) = (BufferId(1), BufferId(2));
    let captured: Captured = Arc::default();
    let cap = captured.clone();
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a, b], |_| vec![vec![10], vec![20]])
        .add_cpu_op(&[b, a], &[], move |ins| {
            *cap.lock().unwrap() = Some(ins.iter().map(|s| s.to_vec()).collect());
            vec![]
        });
    p.register_buffer(a, 1);
    p.register_buffer(b, 1);
    p.build().unwrap().execute(&cpu_runner()).unwrap();
    assert_eq!(
        captured.lock().unwrap().clone(),
        Some(vec![vec![20], vec![10]])
    );
}

#[test]
fn unregistered_cpu_input_is_an_error_not_a_shift() {
    let (a, missing) = (BufferId(1), BufferId(99));
    let ran = Arc::new(Mutex::new(false));
    let ran2 = ran.clone();
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a], |_| vec![vec![7]])
        .add_cpu_op(&[missing, a], &[], move |_| {
            *ran2.lock().unwrap() = true;
            vec![]
        });
    p.register_buffer(a, 1);
    let err = p.build().unwrap().execute(&cpu_runner());
    assert!(err.is_err(), "missing input must be an error, got Ok");
    assert!(!*ran.lock().unwrap(), "op must not run with shifted inputs");
}

#[test]
fn unwritten_cpu_input_is_an_error() {
    let a = BufferId(1);
    let mut p = Pipeline::new().add_cpu_op(&[a], &[], |_| vec![]);
    p.register_buffer(a, 4);
    assert!(p.build().unwrap().execute(&cpu_runner()).is_err());
}

#[test]
fn cpu_op_output_count_mismatch_is_an_error() {
    let (a, b) = (BufferId(1), BufferId(2));
    let mut p = Pipeline::new().add_cpu_op(&[], &[a, b], |_| vec![vec![1]]);
    p.register_buffer(a, 1);
    p.register_buffer(b, 1);
    assert!(p.build().unwrap().execute(&cpu_runner()).is_err());
}

#[test]
fn unregistered_cpu_output_is_an_error() {
    let a = BufferId(1);
    let p = Pipeline::new().add_cpu_op(&[], &[a], |_| vec![vec![1]]);
    assert!(p.build().unwrap().execute(&cpu_runner()).is_err());
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    bytemuck::pod_collect_to_vec(b)
}

/// CPU -> kernel -> CPU -> kernel -> CPU, with the `subtract` shader
/// (`out = a - b`, two read bindings then one read_write).
fn gpu_round_trip(captured: Captured) -> Pipeline {
    let (a, b, c, d, e) = (
        BufferId(1),
        BufferId(2),
        BufferId(3),
        BufferId(4),
        BufferId(5),
    );
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a, b], |_| {
            vec![
                f32_bytes(&[5.0, 6.0, 7.0, 8.0]),
                f32_bytes(&[1.0, 1.0, 1.0, 1.0]),
            ]
        })
        .add_kernel("subtract", &[a, b], &[c], vec![])
        .add_cpu_op(&[c], &[d], |ins| {
            let v: Vec<f32> = bytes_f32(ins[0]).iter().map(|x| x * 10.0).collect();
            vec![f32_bytes(&v)]
        })
        .add_kernel("subtract", &[d, a], &[e], vec![])
        .add_cpu_op(&[c, e], &[], move |ins| {
            *captured.lock().unwrap() = Some(ins.iter().map(|s| s.to_vec()).collect());
            vec![]
        });
    for id in [a, b, c, d, e] {
        p.register_buffer(id, 16);
    }
    p.build().unwrap()
}

fn check_round_trip(captured: &Captured) {
    let got = captured
        .lock()
        .unwrap()
        .clone()
        .expect("final op never ran");
    assert_eq!(
        bytes_f32(&got[0]),
        vec![4.0, 5.0, 6.0, 7.0],
        "kernel 1 output"
    );
    assert_eq!(
        bytes_f32(&got[1]),
        vec![35.0, 44.0, 53.0, 62.0],
        "kernel 2 output"
    );
}

#[test]
fn gpu_kernel_cpu_op_kernel_round_trip() {
    let Some(runner) = gpu_runner() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let captured: Captured = Arc::default();
    gpu_round_trip(captured.clone()).execute(&runner).unwrap();
    check_round_trip(&captured);
}

#[tokio::test(flavor = "multi_thread")]
async fn gpu_kernel_cpu_op_kernel_round_trip_async() {
    let Some(runner) = gpu_runner() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let captured: Captured = Arc::default();
    let mut handle = gpu_round_trip(captured.clone()).execute_async(&runner);
    handle.wait_for_completion().await.unwrap();
    check_round_trip(&captured);
}

#[test]
fn gpu_kernel_with_unregistered_input_is_an_error() {
    let Some(runner) = gpu_runner() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (a, missing, c) = (BufferId(1), BufferId(99), BufferId(3));
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a], |_| vec![f32_bytes(&[1.0; 4])])
        .add_kernel("subtract", &[a, missing], &[c], vec![]);
    p.register_buffer(a, 16);
    p.register_buffer(c, 16);
    assert!(p.build().unwrap().execute(&runner).is_err());
}

#[test]
fn kernel_on_cpu_device_is_an_error() {
    let a = BufferId(1);
    let mut p = Pipeline::new()
        .add_cpu_op(&[], &[a], |_| vec![vec![0; 4]])
        .add_kernel("subtract", &[a, a], &[a], vec![]);
    p.register_buffer(a, 4);
    assert!(p.build().unwrap().execute(&cpu_runner()).is_err());
}
