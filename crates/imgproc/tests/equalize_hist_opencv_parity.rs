//! `histogram_equalization` / `equalize_histogram` against
//! `cv2.equalizeHist` (OpenCV 4.13).
//!
//! OpenCV builds the LUT as `saturate_cast<uchar>(sum * scale)` with
//! `scale = 255.f / (total - hist[first])` in f32, and `saturate_cast`
//! rounds half to even (`cvRound`). Both Rust copies divided first and
//! rounded half away from zero, so every LUT entry landing on .5 came out one
//! level high: in the image below `44 * 255 / 680 = 16.5` gives cv2 16, rust
//! 17. Over 3000 random small images that disagreed with cv2 in 280 of them.
//! Reference outputs below are cv2's, row-major hex.

use cv_imgproc::{equalize_histogram, histogram_equalization};
use image::GrayImage;

fn quantised(w: u32, h: u32, seed: u32, div: u8) -> GrayImage {
    let mut s = seed;
    GrayImage::from_fn(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        image::Luma([(s >> 24) as u8 / div])
    })
}

fn unhex(chunks: &[&str]) -> Vec<u8> {
    let s: String = chunks.concat();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn check(name: &str, img: &GrayImage, expected: &[&str]) {
    let exp = unhex(expected);
    for (which, out) in [
        ("histogram_equalization", histogram_equalization(img)),
        ("equalize_histogram", equalize_histogram(img)),
    ] {
        let n_diff = out
            .as_raw()
            .iter()
            .zip(&exp)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            n_diff, 0,
            "{which}({name}): {n_diff} pixels differ from cv2"
        );
    }
}

#[test]
fn half_level_lut_entry_rounds_to_even() {
    // 1 px of 0, 44 px of 7, 636 px of 200: cv2 maps them to 0, 16, 255.
    let mut px = vec![0u8];
    px.extend(std::iter::repeat_n(7u8, 44));
    px.extend(std::iter::repeat_n(200u8, 636));
    let img = GrayImage::from_raw(681, 1, px).unwrap();
    for out in [histogram_equalization(&img), equalize_histogram(&img)] {
        assert_eq!(out.as_raw()[0], 0);
        assert_eq!(out.as_raw()[1], 16, "44 * 255 / 680 = 16.5 rounds to even");
        assert_eq!(out.as_raw()[680], 255);
    }
}

#[test]
fn quantised_noise_matches_opencv() {
    check("lcg(8) / 7", &quantised(23, 17, 8, 7), LCG8_DIV7);
    check("lcg(10) / 3", &quantised(23, 17, 10, 3), LCG10_DIV3);
    check("lcg(10) / 7", &quantised(23, 17, 10, 7), LCG10_DIV7);
}

#[test]
fn constant_image_is_unchanged() {
    // cv2.equalizeHist(full(.., 77)) == 77.
    let img = GrayImage::from_pixel(5, 4, image::Luma([77]));
    assert_eq!(histogram_equalization(&img).as_raw(), img.as_raw());
    assert_eq!(equalize_histogram(&img).as_raw(), img.as_raw());
}

#[rustfmt::skip]
const LCG8_DIV7: &[&str] = &[
    "440053f9baa5c6f99f09713a7186d444f071c6c00909c0d4f0cbc6c0f017094b1744310077c031ba",
    "1c3a174be6294b66b160b1095ab666e629ff7eac605a29b6178e1cac718631ff29a59f77b122f98e",
    "f922717729294b860000e64bf0ff44ac9ad8665a7ee6e1f9ba6a9af0cbf9538e9ab1ff7e0009537e",
    "4422a500a586c6c63a8e31171cb677c0a58ef0f07766536a77534b3ad4ba77a529ac7e7e3a71d817",
    "e16a9a8e31663a445a1717d4b65aa59af0cbff53094be171e6c6d444ac8e44e1c6ba8ed84409a517",
    "22f0c0ffc0f09fc69a77d4f0a566e15a86b65affcbd453d8533ae1e1319f53e11c0917e1c09f3117",
    "1cf066ff099ae17e9a8e1786ac9a09f922176ae69a9af000173a9a1cd8cbf9e1ac7100d460f971f9",
    "8e608eac171722863ae6bae1604486b19f4b31445a6631f09a7e1c6af98644acb1319a5317f9e129",
    "cbd46053179affe109f931005a3ab644713ad45300b1293af00077869f224bcb6660b64bc6449a8e",
    "8e095a44177e093a2222609a00c6293a17d4d45aa57ed4d4b6771c6a00ac86",
];

#[rustfmt::skip]
const LCG10_DIV3: &[&str] = &[
    "4838ad080e22a8ce9c325e9a676e5635487d6485773522f638b3fb56955842970077f6dc356cf607",
    "90fb3c67293215dc501515c36e4ef6bc1b2922eba5772e00bcdc9522f0d68b07d6f0299a9fd6fb3c",
    "8356b1b8eb07cbf648631b5e1d9c6090d307154c44edb5797da140724cbe63eb9556dcf6322a0e15",
    "29b84c15e87990dcb329ad8bf6009ae6cba18b5b2e6edc3cfec3c1d36c08fe64909ffbe6c500c11b",
    "8be100d14e380ef69ca542edfec857a548ff0e560e1da535be90a5dce8a142d181ad56bc6c7dc332",
    "077d67222a25e050a1def00ebc6ebcb5561b971b50e68022955e408bc1a81de8adf63cdc97fbfbb1",
    "5ed1358722edce3240e044fe72e65879b16cfe50dc6344b8741dc874eb4e56cb85dce6ce0e8b40b5",
    "deb1013c8be1cbcb9fe663a8671529deadd65b568342c10e259aebbc9ac8fb72154c608081904890",
    "1585951b2e157dc51b5b071b326c9560073c80c8d12ef66cf087b3d6bed14895a57263b1297db8a8",
    "ceb81bad83b138000e775b4001e6c59f257780837200ad2ec172074c6c4025",
];

#[rustfmt::skip]
const LCG10_DIV7: &[&str] = &[
    "4638b1090921a7d19b305f9b656d5538467b6585733821fb38b1fb5596594696007bfbde386dfb09",
    "96fb3865283018de4c1818c46d4cfbbd182821eea7733000bdde9621fbd98e00d9ee289ba3d9fb3f",
    "8555b1bde909cbfb4665185f219b5f8ed909184c46eeb57b7ba33f734cbd65e99655d9fb30280918",
    "28bd4c18e97b8edeb528b18efb009be9cba38e59306dd93fffc4c4d96d09ff6596a3fbe9cb00c418",
    "8ee100d14c3809fb9ba73feeffcb59a346ff09550921a338bd8ea3d9e9a33fd180b155bd6d80c430",
    "097b6d212828e155a3defb09bd6dbdb55518961855e98021965f3f8ec4a721e9b1fb38d996fbfbb1",
    "5fd1388521eed1303fe146ff73e9597bb16dff55d96546b57321cb73ee4c55cb85dee9d1188e3fb5",
    "deb100388ee1cbcba3e95fa7651828deb1d959558546c418219beebd9bcbfb73184c5f80808e468e",
    "1885961830187bc418590018306d965f093f80cbd130fb6dee8eb1d9bdd14696a3735fb1287bb5a7",
    "d1bd18b185b13800187b593f00e9c4a3217b80857300b130c473094c6d3f28",
];
