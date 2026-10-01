// Point-cloud sprite renderer.
//
// One quad per point, sized by depth, with the sprite's corners discarded in the
// fragment stage. A cloud is a bag of independent samples, so there is no meshing
// and no lighting; what matters is that a dense cloud does not turn into a solid
// block, which is what the round sprite buys.
//
// `view` is the row-major camera transform. The viewport is already in NDC
// ([-1, 1]) because that is what egui's PaintCallback hands over, so there is no
// projection matrix in this shader.

struct Uniforms {
    view: mat4x4<f32>,
    viewport: vec2<f32>,
    point_radius: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) has_color: f32,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) sprite_uv: vec2<f32>,
};

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0,  1.0),
);

@vertex
fn vs_main(input: VertexIn, @builtin(vertex_index) vi: u32) -> VertexOut {
    let corner = CORNERS[vi];

    // `look_at` on the host produces a **row-major** matrix `m`, where
    // `(m * p)[i] = sum_j m[i][j] * p[j]`.
    //
    // WGSL's `mat4x4<f32>` is **column-major**: `m[i]` is column `i`, so
    // `m[i][j]` is the element at row `j`, column `i`. Summing
    // `uniforms.view[i][j] * world[j]` therefore computes the *transpose*.
    //
    // The original comment here said "row-major multiply, so the matrix is
    // indexed view[col][row]" and then indexed it the other way. The bug is
    // invisible for an identity view - which is what the readback test happened
    // to use - and wrong for every real camera, which silently renders nothing.
    let world = vec4<f32>(input.position, 1.0);
    var eye = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var r = 0u; r < 4u; r = r + 1u) {
        var acc = 0.0;
        for (var c = 0u; c < 4u; c = c + 1u) {
            // Column-major read: element at (row c, column r).
            acc = acc + uniforms.view[c][r] * world[c];
        }
        eye[r] = acc;
    }

    // `eye.w` is the perspective term the host put in the matrix's last row: it
    // is the distance in front of the camera. Using `-eye.z` here instead would
    // have worked only while the matrix was affine - which is why the sprite
    // size looked plausible before the projection existed and wrong after.
    let depth = max(eye.w, 0.01);

    // Perspective-correct radius: a point twice as near covers twice the pixels.
    // 0.02 is an arbitrary near-plane stand-in; it only sets the scale.
    // `point_radius` is a size in *pixels*, so it converts through NDC as
    // radius_px / (viewport_width / 2). The factor here was 0.02, which at a
    // viewport half-width of 1.0 means a point radius of 3 produced 0.03 NDC -
    // under half a pixel. Sprites that small mostly fall between sample points,
    // so a scene of 16,826 points rendered 50 lit pixels.
    //
    // Dividing by depth keeps the perspective-correct behaviour: the same
    // physical size appears the same number of pixels wide whether it is near or
    // far.
    let radius_ndc = uniforms.point_radius / (depth * uniforms.viewport.x);

    // WebGPU requires `z_ndc` in [0, 1], and the host sends only a view matrix -
    // no projection - so this stands in for the projection's depth term.
    //
    // Measured on this machine (Radeon 890M, RTX 5070 Ti, wgpu 28): with
    // `near = 0.01` and `far = 1000`, this scene's points land at z_ndc
    // 0.994-0.997 and **nothing rasterises**; a constant 0.5 renders the whole
    // scene. The cause was not isolated - it is not the [0, 1] range, since
    // 0.994 is inside it, and not f32 precision in the constants, which were
    // checked. Rescaling near/far to the scene does not help either. So this is
    // a conservative mid-range depth rather than a computed projection, with the
    // reason recorded rather than a guess dressed up as a derivation.
    //
    // There is no depth attachment, so depth here only has to stay in range; if
    // one is added later this should become a real projection.
    // `eye` is the pre-divide clip vector: the host's matrix puts the perspective
    // in its last row, so `eye.w` is the depth and `eye.xy` still has to be
    // divided by it. The rasteriser performs that divide, so it must NOT be
    // done here - doing it twice shrinks everything by `w` and puts almost the
    // whole scene off-screen.
    //
    // The sprite offset is the exception: it is a screen-space size, so it has
    // to be added *after* the divide. It therefore goes in as an offset on the
    // clip vector scaled by `w`, which is what keeps a point the same number of
    // pixels across at every depth.
    let offset_x = corner.x * radius_ndc * eye.w / uniforms.viewport.x;
    let offset_y = corner.y * radius_ndc * eye.w / uniforms.viewport.y;

    var out: VertexOut;
    out.clip_position = vec4<f32>(
        eye.x + offset_x,
        eye.y + offset_y,
        0.5 * eye.w,
        eye.w,
    );
    out.sprite_uv = corner;
    // Points with no colour of their own are drawn slightly translucent, which
    // distinguishes an inferred colour from a real one at a glance.
    out.color = vec4<f32>(input.color, select(1.0, 0.85, input.has_color < 0.5));
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    // Round sprite: samples outside the disc are discarded rather than drawn as a
    // square, which is what stops overlapping points from forming a grid.
    let d2 = dot(input.sprite_uv, input.sprite_uv);
    if d2 > 1.0 {
        discard;
    }
    // A soft edge so points blend rather than tile.
    let alpha = input.color.a * (1.0 - smoothstep(0.6, 1.0, d2));
    return vec4<f32>(input.color.rgb, alpha);
}
