//! The root crate is the facade over the workspace libraries.
//!
//! `cv-cli` is a binary, `cv-examples` has no library, and `cv-python` is the
//! PyO3 extension. Every other workspace crate is a module here. Dropping one
//! fails this file at compile time: each name below is a `pub use` of that crate.

#[allow(unused_imports)]
use rust_cv_native::{
    calib3d, core, d3, distributed, dnn, eval, features, geometry2d, hal, imgproc, io,
    localization, math, optimize, photo, plot, pointcloud, registration, rendering, runtime,
    scientific, sfm, signal, slam, video, videoio, viewer,
};

#[test]
fn every_library_crate_is_a_module_of_the_facade() {
    // The imports are the assertion. Twenty-seven library crates: the original
    // eleven, plus plot, math, registration, localization, and the rest that
    // were only reachable by depending on the crate directly.
    let _ = (
        std::mem::size_of::<usize>(),
        stringify!(
            calib3d core d3 distributed dnn eval features geometry2d hal imgproc io localization
            math optimize photo plot pointcloud registration rendering runtime scientific sfm
            signal slam video videoio viewer
        ),
    );
}
