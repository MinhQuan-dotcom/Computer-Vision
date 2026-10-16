import sys
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 9


# =====================================================================
# PHẦN A. CÀI ĐẶT LÕI (tương ứng 4 hàm trong integral_adaptive_thresh.py)
# =====================================================================
def integral_image(img):
    """
    Ảnh tích phân có đệm 0 ở hàng/cột đầu -> kích thước (H+1, W+1).
        I[y, x] = tổng img[0:y, 0:x]
    Dùng float64 để không tràn số khi cộng dồn (đặc biệt với ảnh bình phương).
    """
    img = np.asarray(img, dtype=np.float64)
    I = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    I[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return I


def box_sum(I, y1, x1, y2, x2):
    """
    Tổng các pixel trong hình chữ nhật [y1..y2] x [x1..x2] (bao gồm 2 đầu mút)
    chỉ với 4 phép truy xuất -> O(1), không phụ thuộc kích thước cửa sổ.
        S = I[y2+1,x2+1] - I[y1,x2+1] - I[y2+1,x1] + I[y1,x1]
    y1,x1,y2,x2 có thể là mảng numpy (tính cho cả ảnh một lần).
    """
    return I[y2 + 1, x2 + 1] - I[y1, x2 + 1] - I[y2 + 1, x1] + I[y1, x1]


def local_mean_std(img, radius):
    """
    Trung bình & độ lệch chuẩn cục bộ trong cửa sổ (2r+1)x(2r+1) quanh MỖI pixel.
        mean = S1 / n
        var  = S2 / n - mean^2        (S1: tổng ảnh, S2: tổng ảnh bình phương)
    Xử lý biên: cửa sổ bị cắt (clip) vào ảnh, n = số pixel thực sự nằm trong ảnh.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    I1 = integral_image(img)
    I2 = integral_image(img * img)

    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    y1 = np.clip(ys - radius, 0, H - 1)
    y2 = np.clip(ys + radius, 0, H - 1)
    x1 = np.clip(xs - radius, 0, W - 1)
    x2 = np.clip(xs + radius, 0, W - 1)
    n = (y2 - y1 + 1) * (x2 - x1 + 1)          # (H,1)*(1,W) -> (H,W)

    mean = box_sum(I1, y1, x1, y2, x2) / n
    var = box_sum(I2, y1, x1, y2, x2) / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))        # chặn sai số làm tròn gây var < 0
    return mean, std


def adaptive_threshold(img, radius=15, method="sauvola", t=0.15, k=0.3, R=128.0):
    """
    Nhị phân hóa thích nghi. Trả về uint8: 0 = chữ/đối tượng tối, 255 = nền.

    Bradley-Roth : pixel là chữ nếu  I(x,y) < mean * (1 - t)
    Sauvola      : pixel là chữ nếu  I(x,y) < T,   T = mean * (1 + k * (std/R - 1))
    """
    img = np.asarray(img, dtype=np.float64)
    mean, std = local_mean_std(img, radius)
    if method == "bradley":
        T = mean * (1.0 - t)
    elif method == "sauvola":
        T = mean * (1.0 + k * (std / R - 1.0))
    else:
        raise ValueError("method phải là 'bradley' hoặc 'sauvola'")
    return np.where(img < T, 0, 255).astype(np.uint8)


# =====================================================================
# PHẦN B. CÀI ĐẶT "CHẬM" ĐỂ ĐỐI CHỨNG ĐỘ ĐÚNG VÀ ĐO THỜI GIAN
# =====================================================================
def mean_brute_force(img, radius):
    """O(r^2) / pixel: cộng trực tiếp mọi pixel trong cửa sổ."""
    H, W = img.shape
    f = img.astype(np.float64)
    out = np.empty((H, W))
    for y in range(H):
        for x in range(W):
            win = f[max(0, y - radius):min(H, y + radius + 1),
                    max(0, x - radius):min(W, x + radius + 1)]
            out[y, x] = win.mean()
    return out


def mean_separable(img, radius):
    """O(r) / pixel: tích lũy theo cột, nhưng vẫn phải cộng r..2r+1 cột theo ngang."""
    H, W = img.shape
    f = img.astype(np.float64)
    col = np.zeros((H + 1, W))
    col[1:] = np.cumsum(f, axis=0)
    out = np.empty((H, W))
    for y in range(H):
        y1, y2 = max(0, y - radius), min(H - 1, y + radius)
        for x in range(W):
            x1, x2 = max(0, x - radius), min(W - 1, x + radius)
            s = (col[y2 + 1, x1:x2 + 1] - col[y1, x1:x2 + 1]).sum()
            out[y, x] = s / ((y2 - y1 + 1) * (x2 - x1 + 1))
    return out


def mean_integral(img, radius):
    """O(1) / pixel: dùng ảnh tích phân."""
    return local_mean_std(img, radius)[0]


def self_test():
    """Kiểm chứng: ảnh tích phân cho kết quả y hệt brute-force."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 55)).astype(np.uint8)
    for r in (1, 3, 7, 20):
        m_bf = mean_brute_force(img, r)
        m_ii = mean_integral(img, r)
        s_bf = np.array([[img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].std()
                          for x in range(img.shape[1])] for y in range(img.shape[0])])
        s_ii = local_mean_std(img, r)[1]
        assert np.allclose(m_bf, m_ii, atol=1e-9), f"mean sai tại r={r}"
        assert np.allclose(s_bf, s_ii, atol=1e-6), f"std sai tại r={r}"
    # ảnh toàn hằng -> std = 0, mean = hằng
    const = np.full((30, 30), 77, np.uint8)
    m, s = local_mean_std(const, 5)
    assert np.allclose(m, 77) and np.allclose(s, 0)
    # ảnh 1x1 (biên cực đoan)
    m, s = local_mean_std(np.array([[9]], np.uint8), 4)
    assert m[0, 0] == 9 and s[0, 0] == 0
    print("[OK] self_test: mean/std từ ảnh tích phân khớp brute-force (4 bán kính, biên, ảnh hằng).")


# =====================================================================
# PHẦN C. ẢNH THỬ NGHIỆM
# =====================================================================
def make_text_document(H=300, W=420, seed=42):
    """
    Ảnh 'trang giấy có chữ' bị chiếu sáng không đều (sáng phải, tối trái + bóng đổ).
    Chữ được vẽ bằng matplotlib rồi đọc ngược thành mảng numpy.
    """
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    lines = ["Bai tap lon Thi giac may tinh",
             "Nguong thich nghi bang anh tich phan",
             "Bradley-Roth va Sauvola: T = m(1+k(s/R-1))",
             "Anh tich phan: tong hinh chu nhat O(1)",
             "Chieu sang khong deu lam nguong toan cuc",
             "that bai o vung toi va vung sang.",
             "Nguong cuc bo thich nghi theo tung vung.",
             "ĐH GTVT TP. Ho Chi Minh - 2026"]
    for i, s in enumerate(lines):
        fig.text(0.04, 0.92 - i * 0.115, s, fontsize=10.5, family="DejaVu Sans", va="center")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    gray = rgba[..., :3].astype(np.float64).mean(axis=2)       # 0 (chữ) .. 255 (nền)
    gray = gray[:H, :W]
    H, W = gray.shape

    y, x = np.mgrid[0:H, 0:W]
    illum = 0.30 + 0.70 * (x / W)                               # gradient trái tối -> phải sáng
    cx, cy = 0.30 * W, 0.55 * H                                  # bóng đổ tròn mờ
    shadow = 1.0 - 0.45 * np.exp(-(((x - cx) / 70.0) ** 2 + ((y - cy) / 55.0) ** 2))
    rng = np.random.default_rng(seed)
    img = gray / 255.0 * 235.0 * illum * shadow + rng.normal(0, 3, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_gradient_strokes(H=300, W=400, seed=42):
    """Ảnh giống notebook gốc: nền gradient + các nét ngang tối."""
    y, x = np.mgrid[0:H, 0:W]
    img = 60.0 + (x / float(W)) * 170.0
    rng = np.random.default_rng(seed)
    for row in range(40, H - 40, 25):
        for seg in range(30, W - 30, 45):
            img[row:row + 3, seg:seg + rng.integers(20, 38)] -= 65.0
    img += rng.normal(0, 4, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_flat_image():
    """Nền phẳng rộng + vài nét mảnh + nhiễu nhẹ (để thấy khác biệt Bradley/Sauvola)."""
    rng = np.random.default_rng(1)
    img = np.full((200, 300), 210.0)
    img[80:84, 50:250] = 40
    img[120:124, 80:220] = 40
    img[90:130, 148:152] = 40
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def global_otsu(img):
    """Ngưỡng toàn cục Otsu (tự cài bằng numpy) để làm đối chứng."""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    thr = int(np.nanargmax(sb))
    return np.where(img <= thr, 0, 255).astype(np.uint8), thr


def load_user_image(path):
    """Đọc ảnh thật bằng matplotlib.pyplot.imread, đổi sang xám uint8."""
    a = plt.imread(path)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    if a.dtype != np.uint8:
        a = a * 255.0 if a.max() <= 1.0 else a
    return np.clip(a, 0, 255).astype(np.uint8)


# =====================================================================
# PHẦN D. CÁC THÍ NGHIỆM (mỗi thí nghiệm lưu 1 ảnh PNG)
# =====================================================================
def show(fig, name):
    fig.savefig(name, bbox_inches="tight")
    print("   -> đã lưu", name)


def exp0_integral_visual(doc):
    """Minh họa: ảnh gốc, ảnh tích phân, mean cục bộ, std cục bộ, ngưỡng T."""
    mean, std = local_mean_std(doc, 15)
    T_sauv = mean * (1 + 0.3 * (std / 128.0 - 1))
    fig, ax = plt.subplots(1, 5, figsize=(18, 3.6))
    items = [(doc, "Ảnh gốc"),
             (integral_image(doc), "Ảnh tích phân I (H+1 x W+1)"),
             (mean, "Mean cục bộ (r=15)"),
             (std, "Std cục bộ (r=15)"),
             (T_sauv, "Ngưỡng Sauvola T(x,y)")]
    for a, (im, ti) in zip(ax, items):
        a.imshow(im, cmap="gray"); a.set_title(ti); a.axis("off")
    fig.suptitle("Các đại lượng trung gian của kỹ thuật 2", y=1.03)
    show(fig, "k2_00_trung_gian.png")


def exp1_global_vs_adaptive(doc):
    otsu, thr = global_otsu(doc)
    brad = adaptive_threshold(doc, 20, "bradley", t=0.12)
    sauv = adaptive_threshold(doc, 20, "sauvola", k=0.3)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, (im, ti) in zip(ax, [(doc, "Ảnh gốc (chiếu sáng không đều)"),
                                (otsu, f"Ngưỡng TOÀN CỤC Otsu (T={thr})"),
                                (brad, "Bradley-Roth (r=20, t=0.12)"),
                                (sauv, "Sauvola (r=20, k=0.3)")]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(ti); a.axis("off")
    fig.suptitle("Vì sao cần ngưỡng thích nghi?", y=1.03)
    show(fig, "k2_01_toan_cuc_vs_thich_nghi.png")


def exp2_radius(img):
    radii = [5, 10, 20, 40, 80]
    fig, ax = plt.subplots(2, 5, figsize=(18, 5.6))
    for i, r in enumerate(radii):
        ax[0, i].imshow(adaptive_threshold(img, r, "bradley", t=0.12), cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"Bradley r={r}"); ax[0, i].axis("off")
        ax[1, i].imshow(adaptive_threshold(img, r, "sauvola", k=0.3), cmap="gray", vmin=0, vmax=255)
        ax[1, i].set_title(f"Sauvola r={r}"); ax[1, i].axis("off")
    fig.suptitle("Quét bán kính cửa sổ r", y=1.02)
    show(fig, "k2_02_quet_radius.png")


def exp3_t_k(img):
    ts, ks = [0.05, 0.12, 0.25], [0.15, 0.30, 0.50]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for i, tv in enumerate(ts):
        ax[0, i].imshow(adaptive_threshold(img, 20, "bradley", t=tv), cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"Bradley t={tv}"); ax[0, i].axis("off")
    for i, kv in enumerate(ks):
        ax[1, i].imshow(adaptive_threshold(img, 20, "sauvola", k=kv), cmap="gray", vmin=0, vmax=255)
        ax[1, i].set_title(f"Sauvola k={kv}"); ax[1, i].axis("off")
    fig.suptitle("Quét hệ số t (Bradley) và k (Sauvola), r=20", y=1.0)
    show(fig, "k2_03_quet_t_k.png")


def exp4_flat():
    flat = make_flat_image()
    # Bradley dùng t nhỏ để lộ điểm yếu trên nền phẳng có nhiễu
    brad = adaptive_threshold(flat, 15, "bradley", t=0.02)
    sauv = adaptive_threshold(flat, 15, "sauvola", k=0.3)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, (im, ti) in zip(ax, [(flat, "Nền phẳng rộng + nhiễu"),
                                (brad, "Bradley (t=0.02): sinh điểm nhiễu"),
                                (sauv, "Sauvola (k=0.3): sạch hơn")]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(ti); a.axis("off")
    n_b = int((brad == 0).sum()); n_s = int((sauv == 0).sum())
    fig.suptitle(f"Nền phẳng: số pixel bị gán là 'chữ'  Bradley={n_b}  Sauvola={n_s}", y=1.03)
    show(fig, "k2_04_nen_phang.png")
    print(f"   Bradley gán {n_b} pixel là chữ, Sauvola gán {n_s} pixel (nét chữ thật ≈ 1500 pixel).")


def exp5_timing():
    """
    (a) Đo công bằng chi phí TÍNH MEAN CHO MỘT PIXEL bằng 3 cách, cùng viết bằng
        vòng lặp Python thuần (để thấy đúng số phép cộng, không bị numpy che mất):
          - brute-force : (2r+1)^2 phép cộng      -> O(r^2)
          - separable   : (2r+1)   phép cộng      -> O(r)
          - ảnh tích phân: 4 phép truy xuất       -> O(1)
        Lấy mẫu 300 pixel ở giữa ảnh 512x512 (không bị cắt biên).
    (b) Cả ảnh 1000x1000 bằng hàm vector hóa adaptive_threshold, r tăng đến 400.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (512, 512)).astype(np.uint8)
    f = img.astype(np.float64).tolist()
    col = np.zeros((513, 512)); col[1:] = np.cumsum(img.astype(np.float64), axis=0)
    col = col.tolist()
    I = integral_image(img).tolist()
    pts = [(int(y), int(x)) for y, x in zip(rng.integers(100, 400, 300), rng.integers(100, 400, 300))]

    def t_brute(r):
        t0 = time.perf_counter()
        for y, x in pts:
            s = 0.0
            for j in range(y - r, y + r + 1):
                row = f[j]
                for i in range(x - r, x + r + 1):
                    s += row[i]
            _ = s / ((2 * r + 1) ** 2)
        return time.perf_counter() - t0

    def t_sep(r):
        t0 = time.perf_counter()
        for y, x in pts:
            top, bot = col[y - r], col[y + r + 1]
            s = 0.0
            for i in range(x - r, x + r + 1):
                s += bot[i] - top[i]
            _ = s / ((2 * r + 1) ** 2)
        return time.perf_counter() - t0

    def t_int(r):
        t0 = time.perf_counter()
        for y, x in pts:
            s = I[y + r + 1][x + r + 1] - I[y - r][x + r + 1] - I[y + r + 1][x - r] + I[y - r][x - r]
            _ = s / ((2 * r + 1) ** 2)
        return time.perf_counter() - t0

    radii = [2, 4, 8, 16, 32, 64]
    tb = [t_brute(r) for r in radii]
    ts = [t_sep(r) for r in radii]
    ti = [t_int(r) for r in radii]

    big = np.random.default_rng(1).integers(0, 256, (1000, 1000)).astype(np.uint8)
    radii_big = [5, 20, 50, 100, 200, 400]
    tbig = []
    for r in radii_big:
        t0 = time.perf_counter(); adaptive_threshold(big, r, "sauvola"); tbig.append(time.perf_counter() - t0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].loglog(radii, tb, "ro-", label="O(r²) brute-force")
    ax[0].loglog(radii, ts, "bs-", label="O(r) separable")
    ax[0].loglog(radii, ti, "g^-", label="O(1) ảnh tích phân")
    ax[0].set_title("Mean cục bộ của 300 pixel (thang log-log)"); ax[0].set_xlabel("bán kính r")
    ax[0].set_ylabel("giây"); ax[0].grid(True, which="both", alpha=.3); ax[0].legend()
    ax[1].plot(radii_big, np.array(tbig) * 1000, "g^-")
    ax[1].set_title("Cả ảnh 1000x1000: ảnh tích phân + Sauvola"); ax[1].set_xlabel("bán kính r")
    ax[1].set_ylabel("mili-giây"); ax[1].set_ylim(0, max(tbig) * 1000 * 1.5); ax[1].grid(alpha=.3)
    show(fig, "k2_05_thoi_gian.png")
    print("   r:", radii)
    print("   brute (ms):", [round(v * 1000, 1) for v in tb])
    print("   separable (ms):", [round(v * 1000, 1) for v in ts])
    print("   integral (ms):", [round(v * 1000, 2) for v in ti])
    print("   Cả ảnh 1000x1000, r =", radii_big, "-> ms:", [round(v * 1000, 1) for v in tbig])


def exp6_user_image(path):
    img = load_user_image(path)
    otsu, thr = global_otsu(img)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, (im, ti) in zip(ax, [(img, "Ảnh của bạn (xám)"), (otsu, f"Otsu toàn cục T={thr}"),
                                (adaptive_threshold(img, 15, "bradley", t=0.12), "Bradley r=15 t=0.12"),
                                (adaptive_threshold(img, 15, "sauvola", k=0.3), "Sauvola r=15 k=0.3")]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(ti); a.axis("off")
    show(fig, "k2_06_anh_cua_ban.png")


# =====================================================================
if __name__ == "__main__":
    print("== Kiểm chứng tính đúng đắn ==")
    self_test()

    doc = make_text_document()
    strokes = make_gradient_strokes()

    print("== Thí nghiệm ==")
    exp0_integral_visual(doc)
    exp1_global_vs_adaptive(doc)
    exp2_radius(doc)
    exp3_t_k(strokes)
    exp4_flat()
    exp5_timing()
    if len(sys.argv) > 1:
        exp6_user_image(sys.argv[1])

    plt.show()
    print("Hoàn tất.")
