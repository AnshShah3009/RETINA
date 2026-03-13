fn main() {
    // Python extension modules on macOS must allow undefined Python symbols
    // to be resolved at runtime when loaded by the Python interpreter.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
