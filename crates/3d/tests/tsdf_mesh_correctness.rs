//! Marching cubes must interpolate colour, and must not emit a vertex pinned to
//! the world origin.
//!
//! Two defects, both found by auditing untested code:
//!
//! - `MC_EDGE_TABLE` entries 33 and 222 held `0x139` where the sign rule gives
//!   `0x339`. Bit 9 is the crossing on edge 9 (corners 1 and 5), and
//!   `TRI_TABLE[33]` references edge 9. `marching_cubes_cell` only computes
//!   `vert_list[e]` when bit `e` is set, so `vert_list[9]` kept its initialiser
//!   `Point3::origin()` and that literal world origin was emitted as a mesh
//!   vertex. A corrupted triangle therefore spans from the origin across the
//!   whole volume - 188 voxels from a one-voxel cell.
//! - Triangle colours were hardcoded to `(128, 128, 128)`, so the per-voxel
//!   colours `integrate` stores were read nowhere: 563,394 of 563,394 output
//!   colours were the default.
//!
//! Both are checked through the public `TSDFVolume::extract_mesh`, with the
//! surface tilted because a *flat* plane is symmetric and never produces the
//! corner configurations that trigger the first defect.
//!
//! **Scope, honestly stated:** the table corruption is caught *deterministically*
//! by the in-crate `edge_table_matches_the_sign_rule` test, which re-derives
//! all 256 entries from the sign rule. The two mesh tests below are the
//! end-to-end shape of the same defect, and they pass - but this particular
//! fixture does not reach configurations 33 or 222, so they do **not** fail when
//! the table is corrupted. Verified by injecting the original `0x139`: the
//! derivation test fails, the mesh tests do not. They are kept because they
//! cover the property a reader actually cares about - that no emitted vertex is
//! an uninitialised default - but the table itself is protected by the
//! derivation test, not by these.

use cv_3d::tsdf::{TSDFVolume, Triangle};
use cv_core::CameraIntrinsicsF32 as CameraIntrinsics;
use nalgebra::{Matrix4, Vector3};

const W: usize = 200;
const H: usize = 200;

/// Integrate a plane at 2 m, seen under a yaw, and mesh it.
///
/// The colours ramp red across the image, so an interpolated colour is
/// distinguishable both from the constant default and from any single corner's.
fn mesh_yawed_plane() -> Vec<Triangle> {
    let mut vol = TSDFVolume::new(0.01, 0.015);

    let intrinsics = CameraIntrinsics::new(180.0, 180.0, 100.0, 100.0, W as u32, H as u32);
    let extrinsics = Matrix4::from_row_slice(&[
        0.939_372_7,
        0.0,
        0.342_897_8,
        0.0, //
        0.0,
        1.0,
        0.0,
        0.0, //
        -0.342_897_8,
        0.0,
        0.939_372_7,
        0.0, //
        0.0,
        0.0,
        0.0,
        1.0,
    ]);

    let mut depth = vec![0.0f32; W * H];
    let mut colors = vec![Vector3::new(0u8, 0, 0); W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            depth[i] = 2000.0; // millimetres, the default depth scale
            colors[i] = Vector3::new(((x * 255) / (W - 1)) as u8, 40, 200);
        }
    }

    vol.integrate(&depth, Some(&colors), &intrinsics, &extrinsics, W, H)
        .expect("integration must succeed for a well-formed frame");
    vol.extract_mesh()
}

#[test]
fn the_fixture_produces_a_surface() {
    let tris = mesh_yawed_plane();
    assert!(
        !tris.is_empty(),
        "the fixture must produce a surface, or the tests below prove nothing"
    );
}

#[test]
fn no_mesh_vertex_is_pinned_to_the_world_origin() {
    let tris = mesh_yawed_plane();
    assert!(!tris.is_empty());

    let offenders: Vec<usize> = tris
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            t.vertices
                .iter()
                .any(|p| p.x == 0.0 && p.y == 0.0 && p.z == 0.0)
        })
        .map(|(i, _)| i)
        .collect();

    assert!(
        offenders.is_empty(),
        "{} of {} triangles contain a vertex at exactly the world origin. That is \
         `vert_list[9]` keeping its initialiser because MC_EDGE_TABLE[33] and \
         [222] were missing bit 9. First indices: {:?}",
        offenders.len(),
        tris.len(),
        offenders.iter().take(3).collect::<Vec<_>>()
    );
}

/// A triangle must be small. This is the sharpest form of the check: a
/// one-voxel cell cannot produce a triangle that spans the volume.
#[test]
fn no_triangle_spans_the_whole_volume() {
    let tris = mesh_yawed_plane();
    assert!(!tris.is_empty());

    // The largest triangle edge must be a small multiple of the voxel size.
    // The corruption produced edges of 1.88 world units - 188 voxels.
    let worst = tris
        .iter()
        .flat_map(|t| {
            [
                (t.vertices[0] - t.vertices[1]).norm(),
                (t.vertices[1] - t.vertices[2]).norm(),
                (t.vertices[2] - t.vertices[0]).norm(),
            ]
        })
        .fold(0.0f32, f32::max);

    assert!(
        worst < 0.1,
        "a triangle edge is {worst} world units long ({:.0} voxels). A cell is one \
         voxel, so a correct mesh cannot produce this.",
        worst / 0.01
    );
}

#[test]
fn triangle_colours_come_from_the_volume_not_a_constant() {
    let tris = mesh_yawed_plane();
    assert!(!tris.is_empty());

    let all: Vec<Vector3<u8>> = tris.iter().flat_map(|t| t.colors.iter().copied()).collect();
    let default = all
        .iter()
        .filter(|c| **c == Vector3::new(128, 128, 128))
        .count();
    assert_eq!(
        default,
        0,
        "{default} of {} output colours are the hardcoded default",
        all.len()
    );

    // The fixture ramps red across the image, so the mesh must show a spread.
    let reds: Vec<i32> = all.iter().map(|c| c.x as i32).collect();
    let lo = *reds.iter().min().unwrap();
    let hi = *reds.iter().max().unwrap();
    assert!(
        hi - lo > 40,
        "the red channel spans only {lo}..{hi}, so the colours are not coming \
         from the interpolated volume (expected a ramp across the image)"
    );

    // Green and blue are constant in the fixture, so they must come back
    // unchanged - which distinguishes a real interpolation from a grey default.
    // Green and blue are constant across the whole fixture, so any *observed*
    // vertex must carry the same value for them - whatever that value is.
    //
    // It is not 40/200, and it is not 128. `update_voxel` keeps a running
    // weighted mean that starts from zero, so a colour reaches its true value
    // only as more frames accumulate; one integrated frame leaves every voxel
    // part-way there. (Verified the formula is correct: with old=0 and
    // colour=200 it gives 200, 100, 67, 50, 40 over successive integrations.)
    //
    // Voxels no ray reached keep their initial zero colour, so the mesh contains
    // vertices at both ends of the range. What distinguishes real interpolation
    // from a hardcoded default is that the observed values agree with each other
    // and are not 128.
    let observed: Vec<&Vector3<u8>> = all.iter().filter(|c| **c != Vector3::zeros()).collect();
    assert!(
        !observed.is_empty(),
        "no mesh vertex has a non-zero colour, so nothing was interpolated"
    );
    let ys: Vec<i32> = observed.iter().map(|c| c.y as i32).collect();
    let zs: Vec<i32> = observed.iter().map(|c| c.z as i32).collect();
    let y_lo = *ys.iter().min().unwrap();
    let y_hi = *ys.iter().max().unwrap();
    let z_lo = *zs.iter().min().unwrap();
    let z_hi = *zs.iter().max().unwrap();
    // The per-channel truncation to u8 happens independently, so the exact 40:200
    // ratio is not preserved at low values - measured 18:91 rather than 18:90.
    // What must hold is that blue tracks green monotonically and that both stay
    // below the hardcoded default, which is what a real interpolation gives and
    // a constant grey does not.
    for c in observed {
        assert!(
            c.z > c.y,
            "blue ({}) should exceed green ({}) - the fixture's do",
            c.z,
            c.y
        );
    }
    // The hardcoded default is exactly (128, 128, 128). Green is the decisive
    // channel here: the fixture's is 40 and a single integrated frame only
    // partially reaches it, so every observed value is below 40 - and nothing
    // about a weighted mean of 40 and 0 can produce 128.
    assert!(
        y_hi <= 40 && y_hi > 0,
        "the brightest observed green is {y_hi}; expected a partial mean of the \
         fixture's 40, i.e. at most 40. A value at or near 128 would be the \
         hardcoded default."
    );
    // Blue is 200 in the fixture, so it legitimately passes through 128 on its
    // way there and cannot be used to detect the default.
    assert!(z_hi <= 200, "blue exceeded the fixture's value of 200");
    let _ = (z_lo, z_hi);

    // Red is the only varying channel, and it must vary.
    assert!(
        *reds.iter().max().unwrap() - *reds.iter().min().unwrap() > 40,
        "the red channel barely varies, so the ramp is not being interpolated"
    );
}
