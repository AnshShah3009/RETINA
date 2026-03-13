@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(input_texture);
    
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    let color = textureLoad(input_texture, coords, 0);
    // Standard luminance: 0.299*R + 0.587*G + 0.114*B
    let gray = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    
    textureStore(output_texture, coords, vec4<f32>(gray, gray, gray, 1.0));
}
