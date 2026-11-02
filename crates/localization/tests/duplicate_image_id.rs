//! Adding an image whose id is already in the database replaces it.
//!
//! `DatabaseImage::id` is documented as unique within a database, but
//! `add_image` with an existing id pushed a second image and repointed the id
//! lookup at it. Both copies then went into every index `build` makes: the
//! bag-of-words database held two documents for the id, `len()` and
//! `image_ids()` counted it twice (below: 3 images for 2 ids), and the flat descriptor
//! table carried the old image's descriptors under the id, so a match against
//! one of them resolved its keypoint (and landmark) in the *new* image.

use cv_core::{Descriptor, Descriptors, KeyPoint, Pose};
use cv_localization::{Database, DatabaseImage, Vocabulary};

fn image(id: usize, seed: u8, n: usize) -> DatabaseImage {
    let mut descriptors = Descriptors::new();
    let mut keypoints = Vec::new();
    for i in 0..n {
        let kp = KeyPoint::new(10.0 * i as f64, 5.0 + seed as f64);
        let bytes: Vec<u8> = (0..32u8)
            .map(|b| b.wrapping_mul(seed).wrapping_add(i as u8 * 13))
            .collect();
        descriptors.push(Descriptor::new(bytes, kp));
        keypoints.push(kp);
    }
    DatabaseImage {
        id,
        pose: Some(Pose::identity()),
        keypoints,
        descriptors,
        landmarks: vec![None; n],
    }
}

#[test]
fn re_adding_an_id_replaces_the_image() {
    let pool: Vec<Vec<u8>> = image(0, 3, 8)
        .descriptors
        .iter()
        .chain(image(0, 7, 8).descriptors.iter())
        .map(|d| d.data.clone())
        .collect();
    let mut db = Database::new(Some(Vocabulary::train(&pool, 4, 4, 1)));
    db.add_image(image(1, 5, 4));
    db.add_image(image(7, 3, 6));
    db.add_image(image(7, 7, 5)); // same id, different content
    db.build();

    assert_eq!(db.len(), 2, "one image per id");
    assert_eq!(
        db.image_ids(),
        vec![1, 7],
        "order of first insertion is kept"
    );
    assert_eq!(
        db.image(7).unwrap().descriptors.len(),
        5,
        "the newer image wins"
    );
    assert_eq!(db.bow_index().unwrap().len(), 2, "one BoW document per id");

    // Every flat descriptor of id 7 is a descriptor of the new image.
    let new = image(7, 7, 5);
    let of_seven: Vec<_> = db
        .descriptor_index()
        .iter()
        .enumerate()
        .filter(|(_, r)| r.image_id == 7)
        .collect();
    assert_eq!(of_seven.len(), 5);
    for (flat, r) in of_seven {
        assert_eq!(
            db.descriptors().iter().nth(flat).unwrap().data,
            new.descriptors.iter().nth(r.keypoint).unwrap().data
        );
    }
    assert_eq!(db.descriptor_index().len(), 4 + 5);
}
