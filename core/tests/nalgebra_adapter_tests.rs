use cv_core::float::Float;
use cv_core::nalgebra_adapters::*;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

fn assert_approx_eq<T: Float>(a: T, b: f32, tol: f32) {
    let diff = (a.to_f32() - b).abs();
    assert!(diff < tol, "Expected {} but got {}", b, a.to_f32());
}

fn test_adapters_for_type<T: Float>() {
    let p = Point3::new(1.0, 2.0, 3.0);
    let arr_p: [T; 3] = na_point3_to_array(&p);
    assert_approx_eq(arr_p[0], 1.0, 1e-4);
    assert_approx_eq(arr_p[1], 2.0, 1e-4);
    assert_approx_eq(arr_p[2], 3.0, 1e-4);
    let p_back = array_to_na_point3(&arr_p);
    assert!((p_back.x - 1.0).abs() < 1e-4);
    assert!((p_back.y - 2.0).abs() < 1e-4);
    assert!((p_back.z - 3.0).abs() < 1e-4);

    let v = Vector3::new(4.0, 5.0, 6.0);
    let arr_v: [T; 3] = na_vector3_to_array(&v);
    assert_approx_eq(arr_v[0], 4.0, 1e-4);
    assert_approx_eq(arr_v[1], 5.0, 1e-4);
    assert_approx_eq(arr_v[2], 6.0, 1e-4);
    let v_back = array_to_na_vector3(&arr_v);
    assert!((v_back.x - 4.0).abs() < 1e-4);
    assert!((v_back.y - 5.0).abs() < 1e-4);
    assert!((v_back.z - 6.0).abs() < 1e-4);

    let m3 = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    let arr_m3: [[T; 3]; 3] = na_matrix3_to_array(&m3);
    assert_approx_eq(arr_m3[0][0], 1.0, 1e-4);
    assert_approx_eq(arr_m3[2][2], 9.0, 1e-4);
    let m3_back = array_to_na_matrix3(&arr_m3);
    assert!((m3_back[(0, 0)] - 1.0).abs() < 1e-4);
    assert!((m3_back[(2, 2)] - 9.0).abs() < 1e-4);

    let m4 = Matrix4::new(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    );
    let arr_m4: [[T; 4]; 4] = na_matrix4_to_array(&m4);
    assert_approx_eq(arr_m4[0][0], 1.0, 1e-4);
    assert_approx_eq(arr_m4[3][3], 16.0, 1e-4);
    let m4_back = array_to_na_matrix4(&arr_m4);
    assert!((m4_back[(0, 0)] - 1.0).abs() < 1e-4);
    assert!((m4_back[(3, 3)] - 16.0).abs() < 1e-4);
}

#[test]
fn test_adapters_f32() {
    test_adapters_for_type::<f32>();
}

#[test]
fn test_adapters_f64() {
    test_adapters_for_type::<f64>();
}

#[cfg(feature = "half-precision")]
#[test]
fn test_adapters_f16() {
    test_adapters_for_type::<half::f16>();
}

#[cfg(feature = "half-precision")]
#[test]
fn test_adapters_bf16() {
    test_adapters_for_type::<half::bf16>();
}
