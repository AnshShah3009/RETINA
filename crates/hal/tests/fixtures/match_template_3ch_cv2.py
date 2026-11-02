import cv2, numpy as np

C, H, W, TH, TW = 3, 7, 9, 3, 4


def img_v(c, y, x):
    return (c * 83 + y * 37 + x * 59 + (x * y * 13) % 31 + c * x * 7) % 256


def tmpl_v(c, y, x):
    return (c * 47 + y * 91 + x * 23 + (x + 2 * y + c) % 5 * 17) % 256


img = np.array([[[img_v(c, y, x) for c in range(C)] for x in range(W)] for y in range(H)], np.float32)
tm = np.array([[[tmpl_v(c, y, x) for c in range(C)] for x in range(TW)] for y in range(TH)], np.float32)
M = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
N = ['SqDiff', 'SqDiffNormed', 'Ccorr', 'CcorrNormed', 'Ccoeff', 'CcoeffNormed']
print(f'/// `cv2.matchTemplate` (OpenCV {cv2.__version__}) of the {C}-channel {W}x{H} image and')
print(f'/// {TW}x{TH} template below, float32 HWC, one row per method in `METHODS` order.')
print(f'const CV2_3CH: [[f64; {(H-TH+1)*(W-TW+1)}]; 6] = [')
for m, n in zip(M, N):
    r = cv2.matchTemplate(img, tm, m)
    # same with u8 to make sure the depth does not matter for integers
    print('    // ' + n)
    print('    [' + ', '.join(repr(float(v)) for v in r.ravel()) + '],')
print('];')
# per-channel sum check for the unnormalised ones
for m, n in zip(M[::2], N[::2]):
    full = cv2.matchTemplate(img, tm, m)
    parts = sum(cv2.matchTemplate(img[..., c].copy(), tm[..., c].copy(), m) for c in range(C))
    print('// sum of planes ==', n, np.abs(full - parts).max())
