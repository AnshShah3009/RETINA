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

    // Row-major multiply, so the matrix is indexed view[col][row].
    let world = vec4<f32>(input.position, 1.0);
    var eye = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var r = 0u; r < 4u; r = r + 1u) {
        var acc = 0.0;
        for (var c = 0u; c < 4u; c = c + 1u) {
            acc = acc + uniforms.view[r][c] * world[c];
        }
        eye[r] = acc;
    }

    // The camera looks down -z in view space, so depth is -eye.z.
    let depth = max(-eye.z, 0.01);

    // Perspective-correct radius: a point twice as near covers twice the pixels.
    // 0.02 is an arbitrary near-plane stand-in; it only sets the scale.
    let radius_ndc = uniforms.point_radius * 0.02 / depth;

    var out: VertexOut;
    out.clip_position = vec4<f32>(
        eye.x / eye.w + corner.x * radius_ndc / uniforms.viewport.x,
        eye.y / eye.w + corner.y * radius_ndc / uniforms.viewport.y,
        eye.z / eye.w,
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
