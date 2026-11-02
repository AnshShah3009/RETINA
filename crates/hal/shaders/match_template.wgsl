// `cv::matchTemplate`, every method, planar CHW image and template with
// `params.channels` channels each. One invocation scores one window; every
// term is summed over the channels and the score map has one channel.
//
// `templ_data` is the template as the host prepared it
// (`cpu::template_match::TemplateTerms`): as given for SqDiff* / Ccorr*,
// centred per channel for Ccoeff*. `params.t_norm` is its norm for the
// normalised methods. The normalisation is OpenCV's (`normalise` in
// `cpu::template_match`), in f32.

struct Params {
    img_w: u32,
    img_h: u32,
    templ_w: u32,
    templ_h: u32,
    // 0: SqDiff, 1: SqDiffNormed, 2: Ccorr, 3: CcorrNormed, 4: Ccoeff,
    // 5: CcoeffNormed
    method: u32,
    channels: u32,
    t_norm: f32,
    _pad: u32,
}

// These are f32 buffers, not packed bytes. Declaring them `array<u32>` and
// reassembling four bytes per word reads the low byte of each f32, which is 0x00
// for any whole-number value - so the image and the template both came back as
// mostly zeros and the squared difference was computed against the wrong data.
@group(0) @binding(0) var<storage, read> img_data: array<f32>;
@group(0) @binding(1) var<storage, read> templ_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> score_map: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const FLT_EPSILON: f32 = 1.1920929e-7;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let out_w = params.img_w - params.templ_w + 1u;
    let out_h = params.img_h - params.templ_h + 1u;

    if (x >= out_w || y >= out_h) {
        return;
    }

    let m = params.method;
    let sqdiff = m == 0u || m == 1u;
    let normed = m == 1u || m == 3u || m == 5u;
    let n = f32(params.templ_w * params.templ_h);
    let i_plane = params.img_w * params.img_h;
    let t_plane = params.templ_w * params.templ_h;

    var num = 0.0;
    // Window sum of I^2 over every channel, and for CcoeffNormed the window
    // sum of (I - mean(I_c))^2.
    var wnd_sum2 = 0.0;
    var centred2 = 0.0;
    for (var c = 0u; c < params.channels; c = c + 1u) {
        let ib = c * i_plane;
        let tb = c * t_plane;
        var mean = 0.0;
        if (m == 5u) {
            var s = 0.0;
            for (var j = 0u; j < params.templ_h; j = j + 1u) {
                for (var i = 0u; i < params.templ_w; i = i + 1u) {
                    s = s + img_data[ib + (y + j) * params.img_w + x + i];
                }
            }
            mean = s / n;
        }
        for (var j = 0u; j < params.templ_h; j = j + 1u) {
            for (var i = 0u; i < params.templ_w; i = i + 1u) {
                let iv = img_data[ib + (y + j) * params.img_w + x + i];
                let tv = templ_data[tb + j * params.templ_w + i];
                if (sqdiff) {
                    let d = iv - tv;
                    num = num + d * d;
                } else {
                    num = num + iv * tv;
                }
                if (normed) {
                    wnd_sum2 = wnd_sum2 + iv * iv;
                    let dc = iv - mean;
                    centred2 = centred2 + dc * dc;
                }
            }
        }
    }

    var score = num;
    if (normed) {
        var diff2 = wnd_sum2;
        if (m == 5u) {
            diff2 = centred2;
        }
        diff2 = max(diff2, 0.0);
        var den = 0.0;
        if (diff2 > min(0.5, 10.0 * FLT_EPSILON * wnd_sum2)) {
            den = sqrt(diff2) * params.t_norm;
        }
        if (abs(num) < den) {
            score = num / den;
        } else if (abs(num) < den * 1.125) {
            score = select(-1.0, 1.0, num > 0.0);
        } else {
            score = select(0.0, 1.0, m == 1u);
        }
    }

    score_map[y * out_w + x] = score;
}
