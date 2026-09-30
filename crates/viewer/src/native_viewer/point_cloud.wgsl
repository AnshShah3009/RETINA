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

// Projection onto WebGPU's [0, 1] depth range: z_ndc = A + B / z, with
// `z = -near -> 0` and `z = -far -> 1`. A = B / near and B = 1 / (1/near - 1/far).
const NEAR_PLANE: f32 = 0.01;
const FAR_PLANE: f32 = 1000.0;
const FAR_PLANE_COEFF: f32 = 1.0 / (1.0 / NEAR_PLANE - 1.0 / FAR_PLANE);
const NEAR_PLANE_COEFF: f32 = FAR_PLANE_COEFF / NEAR_PLANE;

// Projection onto WebGPU's [0, 1] depth range: z_ndc = A + B / z, chosen so
// that `z = -near -> 0` and `z = -far -> 1`. With A = B / near and
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

    // The camera looks down -z in view space, so depth is -eye.z.
    let depth = max(-eye.z, 0.01);

    // Perspective-correct radius: a point twice as near covers twice the pixels.
    // 0.02 is an arbitrary near-plane stand-in; it only sets the scale.
    let radius_ndc = uniforms.point_radius * 0.02 / depth;

    // WebGPU, like Vulkan and D3D, requires `z_ndc` in **[0, 1]**. There is no
    // projection matrix anywhere in this pipeline - the host sends only a view
    // matrix - so `eye.z / eye.w` was being handed straight to the rasteriser.
    // With a camera looking down -z that is always negative, i.e. entirely
    // outside the clip volume, so **every primitive was clipped and the window
    // could never have drawn anything at all**.
    //
    // This maps the view frustum onto [0, 1]: `z = -near -> 0`, `z = -far -> 1`.
    let z_ndc = NEAR_PLANE_COEFF + FAR_PLANE_COEFF / eye.z;

    var out: VertexOut;
    out.clip_position = vec4<f32>(
        eye.x / eye.w + corner.x * radius_ndc / uniforms.viewport.x,
        eye.y / eye.w + corner.y * radius_ndc / uniforms.viewport.y,
        z_ndc,
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
