# The viewer: what works, what does not, and where to resume

A working record for an unfinished investigation. Written so that picking it up
later does not require re-deriving everything below.

## What works

`cargo run --example demo` opens a window that renders a striped cube with
perspective, orbit and dolly. The readback test measures **76,315 lit pixels** of
a 320x240 frame.

Five real defects were fixed to get there, in this order:

1. **The background was painted after the render callback**, so an opaque
   rectangle sat on top of the clouds. Black window, no error anywhere.
2. **The camera basis was 180 degrees out.** `look_at` built `z = eye - target`
   then `up x z`, so the eye mapped to its own position and the cloud was drawn
   behind the camera. Verified numerically before the fix.
3. **The matrix multiply was transposed.** The host builds a row-vector
   convention matrix with the translation in the last *row*; WGSL's `mat4x4` is
   column-major. The shader summed `view[r][c] * world[c]`, computing
   M-transpose. Its own comment said "row-major multiply, so the matrix is
   indexed view[col][row]" and then indexed it the other way.
4. **There was no perspective at all.** `look_at`'s last row was `[0,0,0,1]`, so
   `eye.w` was always 1 and nothing divided by anything. Scrolling never changed
   the size of anything.
5. **The sprite radius was 48x too large.** The uniform was called `viewport` and
   carried `(1, 1)` - the NDC half-extent - while the shader used it as a pixel
   count. A radius of 3 became a 192 px disc on a 320 px canvas.

## What does not work

`--image <path>` renders **0 lit pixels**. The demo cube in the same test, same
frame, renders 76,315.

### Ruled out, by measurement

| Hypothesis | How it was checked | Result |
| --- | --- | --- |
| The projection is wrong | projected every point through the real `look_at` | `w` = 2.12..2.68 (positive), ndc x ±0.49, y -0.60..0.75, all in frame |
| `z_ndc` is out of range | computed from the shader's own formula | 0.988..0.991, inside [0, 1] |
| Too many points | capped at 100, 5,000, 20,000, 60,000, 120,000, 185,500 | 0 lit at every count |
| It is the point count | 100 image points vs 100 cube points | image 0, cube 76,315 - so it is the geometry |
| It is the colours | forced every point to neutral grey | still 0 |
| It is the camera | image with the cube's yaw/pitch | still 0 |
| `z_ndc` is NaN or infinite | `0.5 + z_ndc * 0.0` - a NaN would poison the sum | finite |

So: not the count, not the colours, not the camera, not the projection, not the
depth range. The matrix is verified correct end to end and it still renders
nothing.

### Where to resume

**Bisect, do not reason.** Every previous attempt reasoned about a matrix that
was then verified correct, and the bug survived. Substitute geometry instead:

1. Take the working cube path and change **one thing** - replace its positions
   with the image's positions, keeping the cube's colours, camera and count. Note
   the lit count.
2. Then the reverse: the image's path with the cube's positions.
3. Whichever change takes 76,315 to 0 is the cause.

`PC_CAP`, `PC_IMAGE`, `PC_NOCOLOR` and `PC_DUMP` are already wired into the test
for exactly this, and `PC_CUBECAM` selects the cube's camera.

```sh
PC_IMAGE=/tmp/mona.png PC_CAP=100  PC_DUMP=/tmp/f.rgb cargo test -p cv-viewer --lib the_pipeline_actually
python3 -c "from PIL import Image; Image.frombytes('RGB',(320,240),open('/tmp/f.rgb','rb').read()).save('/tmp/f.png')"
```

## The overdraw problem, and why a depth buffer is the real fix

The Mona Lisa dump was a flat olive slab. It is the image - over-drawn.

121,836 points each painting a 4 px sprite is **78x overdraw** on a 320x240
frame. With no depth buffer the last point drawn wins, not the nearest, so a
dense cloud cannot read as a surface at all. Points do not occlude one another,
so a solid object drawn as points needs a depth test to be visible.

A faithful image would need points of radius **0.45 px**, which no rasteriser
draws. Depth is the answer, not point size.

**A depth buffer was attempted and reverted.** With `Depth32Float` attached and
`CompareFunction::Less`, the image rendered 0 lit pixels - and so did
`CompareFunction::Always`, which means the problem is not the comparison. The
depth path is genuinely unfinished; it was removed rather than committed
looking plausible.

Note that the demo cube is *not* affected by the overdraw in practice, because
its points are spread out enough that the surface reads anyway.

## The lesson, which is the most useful part

**Five renderer bugs in a row passed every test I wrote, because every test
inspected a value** - a matrix, a vertex count, a lit-pixel count. A screenshot
settles in one look what a paragraph of numbers does not.

The frame-dump harness exists now (`PC_DUMP`) and should have existed before the
first attempt. Any further visual work here should dump and look at frames
rather than reason about what the shader is doing.

Two tests had **encoded** the bugs they were meant to catch, requiring unit axis
lengths and `w == 1` for a matrix that should have had perspective, and
asserting that `PerPoint` colour mode equalled the height ramp. A test that
pins current behaviour is not automatically a correct specification.