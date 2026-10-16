import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# PHẦN 1 — KỸ THUẬT 1: CLAHE
# Cân bằng lược đồ xám thích nghi có giới hạn tương phản
# ═══════════════════════════════════════════════════════════════

L = 256


def clip_histogram(hist, clip_limit):
    """Cắt và phân phối lại lược đồ. hist: mảng (L,) số nguyên.
    Trả về mảng (L,) đã cắt, tổng không đổi."""
    h = hist.astype(np.int64)
    limit = int(clip_limit)
    if limit < 1:
        limit = 1
    excess = np.sum(np.maximum(h - limit, 0))
    h = np.minimum(h, limit)

    # phân phối đều phần dư cho L mức xám
    h += int(excess // L)
    rem = int(excess % L)
    h[:rem] += 1
    return h


def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    """Trả về mảng (Gy, Gx, L) uint8: bảng tra của từng ô."""
    H, W = img.shape
    Gy, Gx = grid
    luts = np.zeros((Gy, Gx, L), dtype=np.uint8)

    for i in range(Gy):
        y0 = i * H // Gy
        y1 = (i + 1) * H // Gy
        for j in range(Gx):
            x0 = j * W // Gx
            x1 = (j + 1) * W // Gx
            tile = img[y0:y1, x0:x1]

            n_pixels = tile.size
            hist = np.bincount(tile.ravel(), minlength=L).astype(np.int64)

            cl = max(1, int(round(clip_limit * (n_pixels / float(L)))))
            hist = clip_histogram(hist, cl)

            cdf = np.cumsum(hist)
            nz = cdf[cdf > 0]
            cdf_min = nz[0] if nz.size > 0 else 0
            denom = n_pixels - cdf_min
            if denom <= 0:
                lut = np.arange(L, dtype=np.uint8)
            else:
                lut = np.clip(np.round((cdf - cdf_min) / denom * 255.0), 0, 255).astype(np.uint8)
            luts[i, j] = lut

    return luts


def clahe(img, grid=(8, 8), clip_limit=2.0):
    """CLAHE đầy đủ. img: (H, W) uint8. Trả về (H, W) uint8."""
    H, W = img.shape
    Gy, Gx = grid
    luts = build_tile_luts(img, grid, clip_limit)

    if Gy == 1 and Gx == 1:
        return luts[0, 0][img]

    # toạ độ tâm các ô
    cy = np.array([(i * H // Gy + (i + 1) * H // Gy - 1) / 2.0 for i in range(Gy)])
    cx = np.array([(j * W // Gx + (j + 1) * W // Gx - 1) / 2.0 for j in range(Gx)])

    yy, xx = np.mgrid[0:H, 0:W]

    ty = np.searchsorted(cy, yy.ravel(), side='right') - 1
    tx = np.searchsorted(cx, xx.ravel(), side='right') - 1
    ty = np.clip(ty, 0, max(0, Gy - 2))
    tx = np.clip(tx, 0, max(0, Gx - 2))

    dy = cy[ty + 1] - cy[ty] if Gy > 1 else 1.0
    dx = cx[tx + 1] - cx[tx] if Gx > 1 else 1.0
    fy = np.clip((yy.ravel() - cy[ty]) / dy, 0.0, 1.0) if Gy > 1 else np.zeros(H * W)
    fx = np.clip((xx.ravel() - cx[tx]) / dx, 0.0, 1.0) if Gx > 1 else np.zeros(H * W)

    ty1 = np.minimum(ty + 1, Gy - 1)
    tx1 = np.minimum(tx + 1, Gx - 1)
    v = img.ravel()

    # nội suy song tuyến trên 4 giá trị LUT của 4 tâm ô lân cận
    f00 = luts[ty,  tx,  v].astype(np.float64)
    f10 = luts[ty1, tx,  v].astype(np.float64)
    f01 = luts[ty,  tx1, v].astype(np.float64)
    f11 = luts[ty1, tx1, v].astype(np.float64)

    out = (1.0 - fy) * (1.0 - fx) * f00 + \
          (1.0 - fy) * fx         * f01 + \
          fy         * (1.0 - fx) * f10 + \
          fy         * fx         * f11

    return np.round(out).astype(np.uint8).reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# KIỂM THỬ CƠ BẢN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Kiểm thử clip_histogram ===")
    rng = np.random.default_rng(42)
    h_test = rng.integers(0, 500, size=256)
    h_clipped = clip_histogram(h_test, clip_limit=200)
    assert h_clipped.sum() == h_test.sum(), "FAIL: tổng histogram thay đổi!"
    print(f"  sum trước = {h_test.sum()}, sum sau = {h_clipped.sum()}  →  OK")

    print("\n=== Kiểm thử build_tile_luts ===")
    img_test = rng.integers(0, 256, size=(103, 157), dtype=np.uint8)
    luts_test = build_tile_luts(img_test, grid=(8, 8), clip_limit=2.0)
    assert luts_test.shape == (8, 8, 256), f"FAIL: shape = {luts_test.shape}"
    assert luts_test.dtype == np.uint8, f"FAIL: dtype = {luts_test.dtype}"
    print(f"  shape = {luts_test.shape}, dtype = {luts_test.dtype}  →  OK")

    print("\n=== Kiểm thử clahe ===")
    out_test = clahe(img_test, grid=(8, 8), clip_limit=2.0)
    assert out_test.shape == img_test.shape, f"FAIL: shape = {out_test.shape}"
    assert out_test.dtype == np.uint8, f"FAIL: dtype = {out_test.dtype}"
    print(f"  shape = {out_test.shape}, dtype = {out_test.dtype}  →  OK")

    # thêm ảnh chia hết
    img_even = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    out_even = clahe(img_even)
    assert out_even.shape == (256, 256) and out_even.dtype == np.uint8
    print(f"  ảnh 256×256  →  OK")

    print("\n✓ Tất cả kiểm thử đạt.")
