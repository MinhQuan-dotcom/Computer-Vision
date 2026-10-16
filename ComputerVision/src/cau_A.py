import numpy as np


def convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0):
    """Tích chập 2D (có lật nhân), trả về float64.

    mode: 'same' (mặc định, cùng kích thước ảnh), 'valid', 'full'.
    boundary: 'fill' (đệm fillvalue), 'symm' (phản xạ, lặp biên), 'wrap' (tuần hoàn).
    """
    img = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    if img.ndim != 2 or ker.ndim != 2:
        raise ValueError("image và kernel phải là mảng 2 chiều")

    kh, kw = ker.shape
    H, W = img.shape
    kf = ker[::-1, ::-1]  # tích chập = tương quan với nhân đã lật

    if mode == "full":
        py0, py1, px0, px1 = kh - 1, kh - 1, kw - 1, kw - 1
    elif mode == "same":
        py0, px0 = kh // 2, kw // 2
        py1, px1 = kh - 1 - py0, kw - 1 - px0
    elif mode == "valid":
        py0 = py1 = px0 = px1 = 0
    else:
        raise ValueError("mode phải là 'full', 'same' hoặc 'valid'")

    pad = ((py0, py1), (px0, px1))
    if boundary == "fill":
        p = np.pad(img, pad, mode="constant", constant_values=fillvalue)
    elif boundary == "symm":
        p = np.pad(img, pad, mode="symmetric")
    elif boundary == "wrap":
        p = np.pad(img, pad, mode="wrap")
    else:
        raise ValueError("boundary phải là 'fill', 'symm' hoặc 'wrap'")

    oh, ow = p.shape[0] - kh + 1, p.shape[1] - kw + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((max(oh, 0), max(ow, 0)))

    out = np.zeros((oh, ow), dtype=np.float64)
    for dy in range(kh):          # chỉ lặp theo phần tử nhân, không lặp theo pixel
        for dx in range(kw):
            out += kf[dy, dx] * p[dy:dy + oh, dx:dx + ow]
    return out


def gaussian_kernel(sigma, size=None):
    """Nhân Gauss 2D chuẩn hóa (tổng = 1). size mặc định = 2*ceil(3*sigma)+1."""
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def _self_test():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (17, 23)).astype(np.float64)
    ker = rng.random((5, 3))

    # đối chiếu brute-force cho mode 'full' (zero padding)
    kh, kw = ker.shape
    H, W = img.shape
    ref = np.zeros((H + kh - 1, W + kw - 1))
    for y in range(H):
        for x in range(W):
            ref[y:y + kh, x:x + kw] += img[y, x] * ker
    assert np.allclose(convolve2d(img, ker, "full"), ref)

    same = convolve2d(img, ker, "same")
    assert same.shape == img.shape
    assert convolve2d(img, ker, "valid").shape == (H - kh + 1, W - kw + 1)

    # nhân đơn vị giữ nguyên ảnh
    ident = np.zeros((3, 3)); ident[1, 1] = 1
    assert np.allclose(convolve2d(img, ident), img)

    # nhân Gauss bảo toàn tổng ở vùng phẳng
    flat = np.full((20, 20), 7.0)
    g = convolve2d(flat, gaussian_kernel(1.0), boundary="symm")
    assert np.allclose(g, 7.0)
    print("cau_A.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
