import numpy as np 
from cau_A import convolve2d 

def hough_accumulate(edges, delta_deg=1.0, grad_dir=None, grad_delta=10.0):
    """ Tạo mảng tích lũy Hough. Hỗ trợ bỏ phiếu theo hướng gradient."""
    H, W = edges.shape
    # Tính đường chéo tối đa
    D = int(np.ceil(np.sqrt(H**2 + W**2)))
    
    # Tạo mảng góc (thetas) từ 0 đến 180 độ
    thetas_deg = np.arange(0, 180, delta_deg)
    thetas_rad = np.deg2rad(thetas_deg)
    
    # Tạo mảng khoảng cách (rhos)
    rhos = np.arange(-D, D + 1)
    
    # Khởi tạo không gian Hough (A)
    A = np.zeros((len(rhos), len(thetas_deg)), dtype=np.int32)
    
    # Lấy tọa độ các điểm biên (edges > 0)
    y_idxs, x_idxs = np.nonzero(edges)
    
    # Tính sin và cos cho tất cả thetas để tránh tính lại trong vòng lặp
    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)
    
    for i in range(len(y_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        # Bỏ phiếu (Voting)
        if grad_dir is not None:
            # Nếu có hướng gradient, chỉ duyệt các góc quanh hướng đó
            g_dir = grad_dir[y, x]
            # Xử lý góc tuần hoàn (vd: g_dir = 5, grad_delta = 10 -> -5 đến 15 -> 175 đến 15)
            min_angle = (g_dir - grad_delta) % 180
            max_angle = (g_dir + grad_delta) % 180
            
            if min_angle < max_angle:
                valid_mask = (thetas_deg >= min_angle) & (thetas_deg <= max_angle)
            else: # Bắt chéo qua 0/180 độ
                valid_mask = (thetas_deg >= min_angle) | (thetas_deg <= max_angle)
                
            valid_indices = np.where(valid_mask)[0]
        else:
            # Nếu không có hướng gradient, duyệt toàn bộ góc
            valid_indices = np.arange(len(thetas_deg))
            
        for t_idx in valid_indices:
            rho = int(np.round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            rho_idx = rho + D # Dịch chuyển rho để làm index (từ 0)
            A[rho_idx, t_idx] += 1
            
    return A, rhos, thetas_deg

def _gaussian_kernel(sigma):
    """Hàm phụ trợ tạo Gaussian kernel để làm mượt"""
    size = int(6 * sigma + 1)
    if size % 2 == 0: size += 1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

def find_peaks(A, n_peaks, nms_radius=3, smooth_sigma=1.0):
    """Làm mượt không gian Hough và tìm đỉnh bằng NMS."""
    # 1. Làm mượt bằng convolve2d 
    kernel = _gaussian_kernel(smooth_sigma)
    
    
    A_smooth = convolve2d(A.astype(float), kernel) 
    from scipy.signal import convolve2d
    A_smooth = convolve2d(A.astype(float), kernel, mode='same')
    # --------------------------------------------------------------------------
    
    # 2. Thực hiện NMS (Triệt tiêu phi cực đại)
    A_nms = np.zeros_like(A_smooth)
    H, W = A_smooth.shape
    
    # Lấy các điểm có giá trị > 0 (tối ưu hóa)
    y_idx, x_idx = np.nonzero(A_smooth > 0.1 * np.max(A_smooth)) # Chỉ xét điểm tiềm năng
    
    for i in range(len(y_idx)):
        y = y_idx[i]
        x = x_idx[i]
        
        # Xác định giới hạn cửa sổ NMS
        y_min = max(0, y - nms_radius)
        y_max = min(H, y + nms_radius + 1)
        x_min = max(0, x - nms_radius)
        x_max = min(W, x + nms_radius + 1)
        
        # So sánh với giá trị lớn nhất trong lân cận
        local_window = A_smooth[y_min:y_max, x_min:x_max]
        if A_smooth[y, x] == np.max(local_window):
            A_nms[y, x] = A_smooth[y, x]
            
    # 3. Tìm n_peaks lớn nhất
    flat_indices = np.argsort(A_nms.ravel())[::-1][:n_peaks]
    peak_y, peak_x = np.unravel_index(flat_indices, A_nms.shape)
    
    return peak_y, peak_x

def hough_multiscale(edges, n_peaks, coarse_delta=2.0, fine_delta=0.1):
    """Hough đa tỉ lệ: Tìm thô trước, tinh chỉnh sau."""
    H, W = edges.shape
    D = int(np.ceil(np.sqrt(H**2 + W**2)))
    
    # 1. BƯỚC THÔ (COARSE)
    A_coarse, rhos_coarse, thetas_coarse = hough_accumulate(edges, delta_deg=coarse_delta)
    coarse_y, coarse_x = find_peaks(A_coarse, n_peaks, nms_radius=5, smooth_sigma=2.0)
    
    final_rhos = []
    final_thetas = []
    
    # 2. BƯỚC MỊN (FINE)
    y_idxs, x_idxs = np.nonzero(edges)
    
    for i in range(n_peaks):
        # Lấy góc và khoảng cách thô
        t_c = thetas_coarse[coarse_x[i]]
        r_c = rhos_coarse[coarse_y[i]]
        
        # Tạo dải tìm kiếm mịn (ví dụ +- 2 độ)
        t_fine_range = np.arange(t_c - coarse_delta, t_c + coarse_delta + fine_delta, fine_delta)
        # Xử lý góc tuần hoàn (ép về [0, 180))
        t_fine_range = t_fine_range % 180 
        
        # Tạo dải rho mịn (+- vài pixel)
        r_fine_range = np.arange(r_c - 5, r_c + 6)
        
        A_fine = np.zeros((len(r_fine_range), len(t_fine_range)), dtype=np.int32)
        
        t_rad_fine = np.deg2rad(t_fine_range)
        cos_t = np.cos(t_rad_fine)
        sin_t = np.sin(t_rad_fine)
        
        # Quét lại ảnh biên để bình chọn trong dải mịn
        for j in range(len(y_idxs)):
            x = x_idxs[j]
            y = y_idxs[j]
            
            for k in range(len(t_fine_range)):
                rho = int(np.round(x * cos_t[k] + y * sin_t[k]))
                
                # Chỉ tính nếu rho nằm trong dải đang xét
                if r_fine_range[0] <= rho <= r_fine_range[-1]:
                    r_idx = rho - r_fine_range[0] # Dịch index
                    A_fine[r_idx, k] += 1
                    
        # Tìm đỉnh cục bộ trong vùng mịn (không cần NMS phức tạp vì vùng đã nhỏ)
        best_idx = np.argmax(A_fine)
        best_r_idx, best_t_idx = np.unravel_index(best_idx, A_fine.shape)
        
        final_rhos.append(r_fine_range[best_r_idx])
        final_thetas.append(t_fine_range[best_t_idx])
        
    return np.array(final_rhos), np.array(final_thetas)