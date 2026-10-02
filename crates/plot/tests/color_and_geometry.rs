//! `cv_plot::Color` and the depth colour ramp.
//!
//! These are pure functions with hand-computable answers, so the wrong
//! answers are specific: a hex digit parsed with the wrong nibble weight, a
//! ramp whose hue runs the wrong way, or a ramp that is not clamped to
//! `[0, 1]`.
//!
//! What is deliberately NOT tested: `Color::hex` already has an in-crate
//! regression test for malformed input, so this file does not repeat it.

use cv_plot::{Color, PointCloud3D};

/// `Color` derives only `Debug + Clone + Copy` (no `PartialEq`), so colour
/// comparisons go through the RGB triple the tuple fields form.
fn rgb(c: Color) -> (u8, u8, u8) {
    (c.0, c.1, c.2)
}

/// `#rrggbb` parses each byte as `high_nibble * 16 + low_nibble`, and `to_hex`
/// is its inverse. Upper case and a missing `#` are both accepted.
///
/// Wrong answer: treating the pair as two independent nibbles (`0x1234` ->
/// `#112233`), ignoring the leading `#` so `"#123456"` parses one byte late
/// (`#345600`), or `to_hex` printing decimal rather than hex (`rgb(18,52,86)`
/// becoming `"#123456"` vs `"#12,34,86"`).
#[test]
fn hex_round_trips_in_both_cases_and_without_the_hash() {
    for hex in ["#123456", "#000000", "#ffffff", "#abcdef", "#0a1b2c"] {
        assert_eq!(Color::hex(hex).to_hex(), hex, "round trip of {hex}");
    }
    assert_eq!(Color::hex("#ABCDEF").to_hex(), "#abcdef", "upper case");
    assert_eq!(Color::hex("abcdef").to_hex(), "#abcdef", "no leading #");
    // Zero-padded so the output is always 7 characters wide.
    assert_eq!(Color::hex("#000000").to_hex().len(), 7);
    assert_eq!(
        Color::hex("#010203").to_hex(),
        "#010203",
        "leading zeroes kept"
    );
}

/// The named constructors must be the exact RGB triples their names imply, and
/// `rgb`/`Display` must agree with the tuple fields.
///
/// Wrong answer: a swapped channel in `cyan`/`magenta`, `Display` printing
/// `Color(0, 255, 255)`, or `r()/g()/b()` dividing by 256 (so pure white reads
/// 0.996).
#[test]
fn named_colors_and_float_accessors_are_exact() {
    let named = [
        (Color::red(), [255u8, 0, 0]),
        (Color::green(), [0, 255, 0]),
        (Color::blue(), [0, 0, 255]),
        (Color::black(), [0, 0, 0]),
        (Color::white(), [255, 255, 255]),
        (Color::yellow(), [255, 255, 0]),
        (Color::cyan(), [0, 255, 255]),
        (Color::magenta(), [255, 0, 255]),
    ];
    for (color, expected) in named {
        assert_eq!([color.0, color.1, color.2], expected, "{color}");
        assert_eq!(
            rgb(color),
            rgb(Color::rgb(expected[0], expected[1], expected[2])),
            "{color} disagrees with rgb()"
        );
        // `Display` prints the byte channels verbatim - NOT the 0..1 float
        // accessors, which would render pure red as "rgb(1, 0, 0)".
        assert_eq!(
            color.to_string(),
            format!("rgb({}, {}, {})", expected[0], expected[1], expected[2])
        );
        assert_close(color.r(), expected[0] as f64 / 255.0, "r()");
        assert_close(color.g(), expected[1] as f64 / 255.0, "g()");
        assert_close(color.b(), expected[2] as f64 / 255.0, "b()");
    }
}

/// `from_depth` is the documented blue -> cyan -> green -> yellow -> red ramp
/// over `t in [0, 1]`, and clamps outside it.
///
/// The exact values are computable by hand from the source:
/// `r = 0` below `t = 0.5` else `(t - 0.5) * 510`;
/// `g = t * 510` below `0.5` else `(1 - t) * 510`;
/// `b = (0.5 - t) * 510` below `0.5` else `0`.
///
/// Wrong answer: green peaking at `t = 0.5` at 255 (it does, so that one must
/// NOT fail), but red rising from `t = 0` instead of `t = 0.5`, blue not
/// reaching 255 at `t = 0`, the peak green being 128 instead of 255, or
/// `from_depth(-1.0)` / `from_depth(2.0)` panicking on the `as u8` cast
/// (a negative float casts to 0 in Rust, so the real risk is the clamp being
/// removed and `t > 1` producing a channel above 255).
#[test]
fn from_depth_matches_the_documented_ramp_and_clamps() {
    assert_eq!(
        rgb(Color::from_depth(0.0)),
        (0, 0, 255),
        "far point is blue"
    );
    assert_eq!(
        rgb(Color::from_depth(0.5)),
        (0, 255, 0),
        "mid depth is pure green"
    );
    assert_eq!(
        rgb(Color::from_depth(1.0)),
        (255, 0, 0),
        "near point is red"
    );

    // Quarter points, computed by hand from the piecewise formula.
    assert_eq!(rgb(Color::from_depth(0.25)), (0, 127, 127));
    assert_eq!(rgb(Color::from_depth(0.75)), (127, 127, 0));

    // Monotonic: red rises and blue falls with depth, green peaks at 0.5.
    let mut prev = Color::from_depth(0.0);
    for step in 1..=20 {
        let cur = Color::from_depth(step as f64 / 20.0);
        assert!(
            cur.0 >= prev.0,
            "red decreased at t = {}",
            step as f64 / 20.0
        );
        assert!(
            cur.2 <= prev.2,
            "blue increased at t = {}",
            step as f64 / 20.0
        );
        prev = cur;
    }

    // Out-of-range input clamps instead of wrapping.
    assert_eq!(rgb(Color::from_depth(-5.0)), rgb(Color::from_depth(0.0)));
    assert_eq!(rgb(Color::from_depth(5.0)), rgb(Color::from_depth(1.0)));
    assert_eq!(
        Color::from_depth(f64::NAN).0,
        0,
        "NaN must not produce a huge channel"
    );
}

/// `PointCloud3D::colorize_by_depth` must map the cloud's own z range onto the
/// ramp: the nearest point is red, the farthest blue, and the extremes match
/// `from_depth(0)` / `from_depth(1)` exactly.
///
/// Wrong answer: using a fixed `[0, 1]` z range instead of the cloud's (so a
/// cloud in `z = 5..10` comes out all red), inverting the ramp (near = blue),
/// or normalising against `abs`/max instead of the span.
#[test]
fn colorize_by_depth_spans_the_clouds_own_z_range() {
    let mut pc = PointCloud3D::new("depth");
    pc.add_points(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[-7.5, 0.0, 12.5]);
    pc.colorize_by_depth();

    assert_eq!(
        rgb(pc.points[0].color),
        rgb(Color::from_depth(0.0)),
        "the minimum-z point must be the ramp's cold end"
    );
    assert_eq!(
        rgb(pc.points[2].color),
        rgb(Color::from_depth(1.0)),
        "the maximum-z point must be the ramp's hot end"
    );
    assert!(
        pc.points[2].color.0 > pc.points[0].color.0,
        "red must increase with depth"
    );
}

/// A cloud whose points all share one z has a zero span; the ramp must not
/// divide by it.
///
/// Wrong answer: `0/0` producing a NaN cast to 0 and colouring every point as
/// pure blue (the ramp's cold end) instead of the neutral mid-ramp value, or a
/// panic from a float-to-integer cast of `NaN`.
#[test]
fn colorize_by_depth_on_a_flat_cloud_is_the_midpoint_not_nan() {
    let mut pc = PointCloud3D::new("flat");
    pc.add_points(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[4.0, 4.0, 4.0]);
    pc.colorize_by_depth();

    let expected = rgb(Color::from_depth(0.5));
    for p in &pc.points {
        assert_eq!(rgb(p.color), expected, "a zero z span must map to t = 0.5");
    }
}

/// `add_points` truncates to the shortest input array; a mismatch between x,
/// y and z is the classic way to index past the end.
///
/// Wrong answer: `points.len()` following the longest array (reading past the
/// end of the shorter one, or emitting default zeros for the extra entries), a
/// panic on a ragged input, or the count following `x.len()` alone.
#[test]
fn ragged_add_points_truncates_to_the_shortest_array() {
    let mut pc = PointCloud3D::new("ragged");
    let before_with_normals = 0usize;

    pc.add_points(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0], &[9.0, 8.0, 7.0]);
    assert_eq!(pc.points.len(), 2, "must stop at the shortest of x/y/z");
    assert_eq!(
        (pc.points[0].x, pc.points[0].y, pc.points[0].z),
        (0.0, 0.0, 9.0)
    );
    assert_eq!(
        (pc.points[1].x, pc.points[1].y, pc.points[1].z),
        (1.0, 1.0, 8.0)
    );

    // The normals and the fully-populated variants truncate the same way, and
    // each call appends to whatever the cloud already holds rather than
    // replacing it. `before_with_normals` is 0 because a fresh cloud is empty.
    assert_eq!(before_with_normals, 0, "a fresh cloud starts empty");
    pc.add_points_with_normals(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[1.0, 1.0, 9.9],
        &[0.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(
        pc.points.len(),
        5,
        "2 already present + 3 (the short nx array bounds the batch)"
    );
    // Index pairing is preserved: the point at x = 3.0 was skipped (its nx was
    // cut off) and must not appear at all. A bug that followed `x.len()`
    // instead of the shortest array would append it with a default normal.
    assert_eq!(pc.points[4].x, 2.0, "only x = 0, 1, 2 were added");
    assert_eq!(pc.points[4].nx, Some(9.9), "index 2 pairs with nx[2]");

    let colors = [Color::red(), Color::green()];
    pc.add_points_with_all(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &colors,
    );
    assert_eq!(pc.points.len(), 7, "5 already present + 2 (only 2 colours)");
    assert_eq!(rgb(pc.points[6].color), rgb(Color::green()));
    assert_eq!(
        pc.points[6].nx,
        Some(0.0),
        "normal was supplied for this point"
    );

    pc.add_colored_points(&[0.0, 1.0], &[0.0], &[0.0, 1.0], &colors);
    assert_eq!(pc.points.len(), 8, "y was the shortest array");
    assert_eq!(rgb(pc.points[7].color), rgb(Color::red()));

    // An empty array adds nothing and leaves the cloud intact.
    pc.add_points(&[], &[1.0, 2.0], &[3.0, 4.0]);
    assert_eq!(pc.points.len(), 8);
}

/// `bounding_box` returns `(min_x, max_x, min_y, max_y, min_z, max_z)`, and the
/// empty cloud has a documented degenerate box rather than infinities.
///
/// Wrong answer: the axis order being `(x, y, z, min, max, ...)` so the
/// extrema land in the wrong slots, min and max swapped, infinities for an
/// empty cloud (which then poison the projection in `Plot3D::to_svg`), or a
/// single point collapsing the box to `(0, 0, 0, 0, 0, 0)`.
#[test]
fn bounding_box_reports_per_axis_extrema_in_slot_order() {
    let empty = PointCloud3D::new("empty");
    assert_eq!(
        empty.bounding_box(),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        "the empty cloud must use the documented degenerate box"
    );
    let (ex_min_x, ex_max_x, ..) = empty.bounding_box();
    assert!(
        ex_min_x.is_finite() && ex_max_x.is_finite(),
        "empty box leaked an infinity"
    );

    let mut pc = PointCloud3D::new("box");
    pc.add_points(&[-1.0, 4.0, 0.0], &[10.0, -2.0, 0.0], &[5.0, 5.0, -3.0]);
    let (min_x, max_x, min_y, max_y, min_z, max_z) = pc.bounding_box();
    assert_eq!((min_x, max_x), (-1.0, 4.0), "x extrema");
    assert_eq!((min_y, max_y), (-2.0, 10.0), "y extrema");
    assert_eq!((min_z, max_z), (-3.0, 5.0), "z extrema");

    // A single point is a zero-extent box at that point, not the origin.
    let mut one = PointCloud3D::new("one");
    one.add_point(7.0, 8.0, 9.0);
    assert_eq!(one.bounding_box(), (7.0, 7.0, 8.0, 8.0, 9.0, 9.0));
}

/// `Point3D` constructors must agree on every field they set, and `with_size`
/// must be the only thing that changes the size.
///
/// Wrong answer: `with_color` also setting a normal (or `with_normal` also
/// overriding the default colour), a default size other than 3.0, or
/// `with_size` consuming `self` without applying the value.
#[test]
fn point_constructors_set_exactly_their_fields() {
    let plain = cv_plot::Point3D::new(1.0, 2.0, 3.0);
    assert_eq!((plain.nx, plain.ny, plain.nz), (None, None, None));
    assert_eq!(rgb(plain.color), rgb(Color::blue()), "default point colour");
    assert_close(plain.size, 3.0, "default point size");

    let with_normal = cv_plot::Point3D::with_normal(1.0, 2.0, 3.0, 0.0, 0.0, 1.0);
    assert_eq!(
        (with_normal.nx, with_normal.ny, with_normal.nz),
        (Some(0.0), Some(0.0), Some(1.0))
    );
    assert_eq!(
        rgb(with_normal.color),
        rgb(Color::blue()),
        "a normal must not change the colour"
    );

    let with_color = cv_plot::Point3D::with_color(1.0, 2.0, 3.0, Color::red());
    assert_eq!(rgb(with_color.color), rgb(Color::red()));
    assert_eq!(
        (with_color.nx, with_color.ny, with_color.nz),
        (None, None, None)
    );

    let all = cv_plot::Point3D::with_all(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, Color::green());
    assert_eq!(rgb(all.color), rgb(Color::green()));
    assert_eq!((all.nx, all.ny, all.nz), (Some(1.0), Some(0.0), Some(0.0)));

    let resized = plain.with_size(11.0);
    assert_close(resized.size, 11.0, "with_size must apply");
    assert_eq!(
        (resized.x, resized.y, resized.z),
        (1.0, 2.0, 3.0),
        "with_size must not move the point"
    );
}

/// `PointCloud3D::color` overrides every existing point, not just the ones
/// added later, and the cloud keeps its own default colour.
///
/// Wrong answer: only points added after the `color()` call being recoloured
/// (a clone of the vector instead of a rebuild), or `color()` dropping the
/// cloud-level default.
#[test]
fn color_recolours_every_point_including_earlier_ones() {
    let mut pc = PointCloud3D::new("c");
    pc.add_point(0.0, 0.0, 0.0);
    pc.add_point(1.0, 1.0, 1.0);
    let recoloured = pc.clone().color(Color::yellow());
    assert_eq!(
        rgb(recoloured.color),
        rgb(Color::yellow()),
        "cloud default colour"
    );
    assert!(
        recoloured
            .points
            .iter()
            .all(|p| rgb(p.color) == rgb(Color::yellow())),
        "every pre-existing point must be recoloured"
    );
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "{what}: expected {expected}, got {actual}"
    );
}
