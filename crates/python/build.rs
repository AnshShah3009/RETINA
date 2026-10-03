fn main() {
    // Python extension modules on macOS must allow undefined Python symbols
    // to be resolved at runtime when loaded by the Python interpreter.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }

    // A test binary under `crates/python/tests/` links against this crate as an
    // rlib, and PyO3's `#[pyclass]` machinery references CPython symbols
    // (`PyExc_ValueError`, `PyErr_Print`, ...) that a plain Rust link does not
    // resolve. The cdylib never needed this - the interpreter provides the symbols
    // at load time - but an integration test is an ordinary executable, so it has to
    // link libpython itself.
    //
    // Without this, `binding_contract.rs` fails with
    //     ld.lld: error: undefined symbol: PyExc_ValueError
    // which is a property of the test harness, not of the binding code.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        if let Ok(dir) = std::env::var("PYTHON_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=python3.13");
        } else if let Ok(out) = std::process::Command::new("python3")
            .args([
                "-c",
                "import sysconfig;print(sysconfig.get_config_var('LIBDIR') or '')",
            ])
            .output()
        {
            if out.status.success() {
                let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !dir.is_empty() && std::path::Path::new(&dir).is_dir() {
                    println!("cargo:rustc-link-search=native={dir}");
                    println!("cargo:rustc-link-lib=python3.13");
                    // The interpreter's shared object lives in that same directory
                    // (a conda prefix, outside the default loader path), so the
                    // test binary exits 127 with
                    //     error while loading shared libraries: libpython3.13.so.1.0
                    // without an rpath even though it linked successfully.
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
                }
            }
        }
    }
}
