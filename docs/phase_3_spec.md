# Specification: Phase 3 — Training Engine và Vanilla 3DGS Baseline Candidate

Dự án này nhận dữ liệu và mô hình hình học từ Phase 2 (thông qua `SceneManifest` và `SceneDataset`) để huấn luyện và render ảnh Novel View Synthesis. Tài liệu này đặc tả kiến trúc hệ thống, các phép toán cốt lõi, baseline B0, và lộ trình triển khai chi tiết cho Phase 3.

---

## 1. Mục tiêu & Phạm vi (Goal & Scope)

Mục tiêu của Phase 3 là xây dựng một **Training Engine hoàn chỉnh, đúng đắn, có tính tái lập và an toàn tài nguyên**, có khả năng tối ưu hóa các tham số Gaussian trên từng trạm phát sóng BTS đơn lẻ.

```
[ Khởi tạo Gaussian từ SfM ] 
           │
           ▼
[ Render khả vi (gsplat) ] ───► [ So sánh ảnh render & Ground Truth ]
           │                                      │
           ▼                                      ▼
[ Cập nhật Adam Optimizer ] ◄────────── [ Tính Loss (L1 + SSIM) ]
           │
           ▼
[ Adaptive Density Control ] ───► [ Checkpoint / Resume ]
```

* **Yêu cầu cốt lõi**: Khóa cấu hình baseline **B0** để so sánh và kiểm chứng tính hội tụ trên các scene thật trước khi chuyển sang Phase 4 (benchmark quy mô lớn).
* **Đơn vị xử lý**: Huấn luyện và suy luận độc lập trên từng scene (per-scene). Không yêu cầu học chuyển giao (cross-scene generalization).

---

## 2. Đặc tả dữ liệu đầu vào & đầu ra (Data I/O Contract)

### 2.1. Phân chia dữ liệu đầu vào (Input Constraints)
Trainer chỉ được phép truy cập và tính toán dựa trên các thành phần sau:
* Dữ liệu huấn luyện: `train_images`, `train_world_to_camera` (poses), `train_intrinsics`, `train_distortion`.
* Dữ liệu khởi tạo: Các điểm 3D sparse có hỗ trợ (`sparse_points` và màu tương ứng `sparse_colors` lọc theo `train_support`).
* Phép chuẩn hóa: `normalization_transform` và `inverse_normalization_transform` của scene.
* Mặt nạ pixel hợp lệ: `valid_masks` (tính từ quá trình undistortion và resize).
* Cấu hình thiết lập baseline B0.

> [!WARNING]
> **Nghiêm cấm tuyệt đối**: Sử dụng dữ liệu test poses làm tín hiệu tối ưu pose hay tính gradient; sử dụng ảnh test ground-truth cho mục tiêu tối ưu hoặc early-stop tự động.

### 2.2. Đặc tả dữ liệu đầu ra (Output Artifacts)
Mỗi tiến trình huấn luyện phải lưu trữ kết quả trong cấu trúc sau:
```text
runs/<scene>/<run_id>/
├── config.yaml               # Snapshot cấu hình chạy
├── environment.json          # Thông tin môi trường (versions, GPU)
├── manifest_hash.json        # Mã băm SHA256 của manifest thô
├── checkpoints/              # Thư mục lưu checkpoint đầy đủ
│   ├── step_000003000.pt
│   └── ...
├── train_previews/           # Ảnh preview các góc train cố định
├── validation_renders/       # Ảnh render góc validation phục vụ giám sát
├── metrics.jsonl             # Ghi log loss và metrics theo định dạng JSON Lines
├── timing.json               # Đo thời gian chi tiết từng phân đoạn step
└── summary.json              # Tổng hợp điểm số, VRAM tối đa, tổng thời gian
```

---

## 3. Cấu hình Baseline B0

Cấu hình Baseline B0 phục vụ Phase 3 được thiết lập cố định như sau:

| Thành phần | Đặc tả kỹ thuật cho B0 |
| :--- | :--- |
| **Representation** | Anisotropic 3D Gaussians (Ma trận covariance phân tách) |
| **Initialization** | Khởi tạo từ các điểm SfM sparse có train support |
| **Renderer** | `gsplat.rasterization` |
| **Camera Domain** | Undistorted pinhole (xử lý méo ảnh ở dataset domain trước) |
| **Batch Size** | 1 camera (chọn ngẫu nhiên uniform per step) |
| **Appearance** | Hệ số Spherical Harmonics tăng dần (coarse-to-fine: degree 0 → 3) |
| **Loss Function** | $L = 0.8 L_1 + 0.2 (1 - \text{SSIM})$ |
| **Optimizer** | Adam với các tham số nhóm riêng biệt (Parameter Group) |
| **Density Control** | `gsplat.DefaultStrategy` (Pruning & Densification) |
| **Training Steps** | Tối đa 30,000 steps |
| **Densification Schedule** | Bắt đầu từ step 500, lặp lại mỗi 100 steps, dừng ở step 15,000 |
| **Pose Optimization** | **TẮT** |
| **Appearance Embedding** | **TẮT** |
| **Depth Loss / LPIPS Loss**| **TẮT** |
| **Anti-Aliasing / MCMC / AMP** | **TẮT** |
| **Random Background** | **TẮT** (Sử dụng màu nền cố định/đen) |
| **Packed Mode** | **BẬT** (Tiết kiệm VRAM trên card L4) |
| **Seed** | Cố định và deterministic |

---

## 4. Cơ sở toán học (Mathematical Foundations)

### 4.1. Biểu diễn của một Gaussian 3D
Mỗi Gaussian $G_i$ được đặc tả bởi 5 thuộc tính:
$$G_i = \left( \mu_i, s_i, q_i, o_i, \beta_i \right)$$

1. **Mean ($\mu_i \in \mathbb{R}^3$)**: Tọa độ tâm Gaussian trong hệ tọa độ thế giới đã chuẩn hóa (normalized world coordinates).
2. **Scale ($s_i \in \mathbb{R}^3$)**: Log-scale. Scale thực tế được tính qua phép mũ để đảm bảo luôn dương:
   $$\hat{s}_i = \exp(s_i), \quad S_i = \operatorname{diag}(\hat{s}_i)$$
3. **Rotation ($q_i \in \mathbb{R}^4$)**: Quaternion đơn vị biểu diễn định hướng:
   $$\hat{q}_i = \frac{q_i}{\|q_i\|}, \quad R_i = R(\hat{q}_i)$$
4. **Covariance ($\Sigma_i$)**: Ma trận covariance 3D được tính từ rotation và scale:
   $$\Sigma_i = R_i S_i S_i^T R_i^T$$
5. **Opacity ($o_i \in \mathbb{R}$)**: Opacity logit. Hệ số truyền sáng thực tế $\alpha_i \in (0, 1)$ được tính qua hàm sigmoid:
   $$\alpha_i = \sigma(o_i) = \frac{1}{1 + \exp(-o_i)}$$
6. **Spherical Harmonics ($\beta_{i,l,m}$)**: Biểu diễn màu sắc phụ thuộc hướng nhìn. Để ổn định hình học ban đầu, B0 sử dụng lịch trình coarse-to-fine:
   $$L(t) = \min\left( \left\lfloor \frac{t}{1000} \right\rfloor, 3 \right)$$

### 4.2. Chuẩn hóa không gian (Scene Normalization)
Để đảm bảo thống nhất learning rate và densification threshold giữa các scene có tỷ lệ khác nhau, các tọa độ điểm $P$ và tâm camera $C$ được chuẩn hóa dựa trên scale $s$ và tâm $\mu$ tính từ Phase 2:
$$P_n = s(P_w - \mu)$$
$$T_{c2w}^n = \begin{bmatrix} R_{cw} & s(C_w - \mu) \\ 0 & 1 \end{bmatrix}$$
* **Ràng buộc kiểm tra (Invariant)**: Phép chiếu ảnh (projection) trước và sau chuẩn hóa phải bảo toàn:
  $$\pi(K, T_{w2c}, X_w) \approx \pi(K, T_{w2c}^n, X_n)$$

### 4.3. Khởi tạo tham số từ SfM (Initialization)
* **Mean ($\mu_i^{(0)}$)**: Khởi tạo trực tiếp bằng tọa độ điểm sparse normalized $P_i^n$.
* **Scale ($s_i^{(0)}$)**: Tính khoảng cách Euclidean trung bình $d_i$ từ điểm $i$ tới $3$ điểm lân cận gần nhất (3-Nearest Neighbors):
   $$\hat{s}_i^{(0)} = d_i \cdot \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}, \quad s_i^{(0)} = \log(\hat{s}_i^{(0)})$$
* **Rotation ($q_i^{(0)}$)**: Khởi tạo dạng Quaternion đơn vị tĩnh: $\begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix}$.
* **Opacity ($o_i^{(0)}$)**: Khởi tạo opacity thực bằng $0.1$, tương đương giá trị logit $\approx -2.1972$.
* **Color ($\beta_i^{(0)}$)**: Gán giá trị màu RGB của điểm SfM vào hệ số SH bậc 0. Các bậc cao hơn gán bằng 0.

### 4.4. Chi chiếu hình ảnh & Alpha Compositing
* **Covariance 2D ($\Sigma_i^{2D}$)**: Chiếu covariance 3D vào mặt phẳng ảnh sử dụng ma trận Jacobian chiếu $J_i$ tại tâm camera:
  $$\Sigma_i^{2D} = J_i R_{wc} \Sigma_i R_{wc}^T J_i^T$$
* **Alpha Blending**: Màu sắc tại mỗi pixel $p$ được tính bằng cách tổng hợp màu từ các Gaussian phân lớp theo độ sâu (từ gần đến xa):
  $$C(p) = \sum_i T_i(p) a_i(p) c_i(d) + T_{\text{end}}(p) C_{\text{bg}}$$
  Trong đó độ truyền quang tích lũy là $T_i(p) = \prod_{j<i} (1 - a_j(p))$ và alpha cục bộ là $a_i(p) = \alpha_i \exp\left( -\frac{1}{2} (p-m_i)^T (\Sigma_i^{2D})^{-1} (p-m_i) \right)$.

---

## 5. Đặc tả Loss Function & Optimizer

### 5.1. Loss Function
Hàm loss kết hợp L1 cục bộ và cấu trúc SSIM toàn cục:
$$L = (1 - \lambda) L_1 + \lambda L_{\text{DSSIM}} \quad \text{với } \lambda = 0.2$$

* **Masked L1**: Chỉ tính toán sai lệch trên các pixel nằm trong `valid_mask` $M(p)$:
  $$L_1 = \frac{\sum_p M(p) |C_{\text{pred}}(p) - C_{\text{gt}}(p)|}{3 \sum_p M(p)}$$
* **Masked SSIM ($L_{\text{DSSIM}}$)**: Áp dụng thuật toán xói mòn (erosion) với kernel kích thước $11 \times 11$ lên mặt nạ hợp lệ để tạo mặt nạ vùng biên an toàn $M_{\text{window}}$:
  $$M_{\text{window}} = \operatorname{Erode}(M, 11 \times 11)$$
  $$L_{\text{DSSIM}} = 1 - \frac{\sum_p M_{\text{window}}(p) \cdot \text{SSIM}_p}{\sum_p M_{\text{window}}(p)}$$

### 5.2. Nhóm tối ưu (Optimizer Parameter Groups)
Sử dụng bộ tối ưu Adam độc lập cho từng nhóm tham số với các tốc độ học (learning rates) sau:

| Tên nhóm tham số | Khởi tạo Learning Rate (LR) | Lịch trình LR (LR Schedule) |
| :--- | :--- | :--- |
| **Mean (Positions)** | $1.6 \times 10^{-4}$ | Exponential decay xuống $1\%$ sau $30,000$ steps |
| **Scale** | $5.0 \times 10^{-3}$ | Cố định |
| **Opacity** | $5.0 \times 10^{-2}$ | Cố định |
| **Rotation (Quaternion)** | $1.0 \times 10^{-3}$ | Cố định |
| **SH coefficients bậc 0** | $2.5 \times 10^{-3}$ | Cố định |
| **SH coefficients bậc > 0**| $1.25 \times 10^{-4}$ | Cố định |

---

## 6. Kiểm soát mật độ thích ứng (Adaptive Density Control)

B0 sử dụng lớp quản lý `gsplat.DefaultStrategy` để thực hiện các hành động densification và pruning dựa trên gradient tích lũy trong không gian màn hình $\nabla_{m_i} L$.

### 6.1. Quy tắc quyết định mật độ
* **Tích lũy gradient**: Tính trung bình chuẩn gradient của projected mean $g_i$.
* **Duplicate**: Nếu $g_i > 0.0002$ và Gaussian có kích thước không quá 1% kích thước scene ($\frac{\max(\hat{s}_i)}{\text{scene\_scale}} \le 0.01$).
* **Split**: Nếu $g_i > 0.0002$ và Gaussian lớn ($\frac{\max(\hat{s}_i)}{\text{scene\_scale}} > 0.01$). Chia Gaussian lớn làm hai với hướng ngẫu nhiên và phân rã scale theo hệ số $1.6$.
* **Pruning**: Chỉ loại bỏ các Gaussian có độ mờ thực tế thấp: $\alpha_i < 0.005$. Scale-based pruning của `DefaultStrategy` được vô hiệu hóa trong B0.
* **Opacity Reset**: Mỗi $3,000$ steps, thiết lập lại toàn bộ opacity của Gaussian về giá trị cực thấp để làm sạch các điểm ảo và cho phép các vùng bị che khuất tối ưu lại.

### 6.2. Lịch trình Density Control
* Step dùng chỉ số **1-based**; không gọi strategy với step 0.
* Bắt đầu densification tại step **500**.
* Thực hiện densification mỗi **100** steps.
* Lần densification cuối là step **14,900** và dừng tại step **15,000**.

---

## 7. Khả năng phục hồi & Lưu trữ an toàn (Resiliency & Checkpoints)

* **Nội dung lưu trữ**: Mỗi checkpoint phải bao gồm toàn bộ trạng thái huấn luyện: bước step hiện tại, tham số Gaussian, trạng thái optimizer (moments), trạng thái LR scheduler, trạng thái tích lũy của `DefaultStrategy`, bậc SH hiện tại, và trạng thái RNG (Python, NumPy, PyTorch CPU & CUDA).
* **Xác thực Manifest (Hash Validation)**: Checkpoint phải đi kèm mã hash SHA256 của Manifest. Khi khôi phục huấn luyện (resume), nếu mã hash không trùng khớp, hệ thống phải báo lỗi và dừng lập tức.
* **Ghi file an toàn (Atomic Write)**: Ghi checkpoint ra file tạm `.pt.tmp` trước, sau khi ghi xong mới tiến hành đổi tên thành `.pt` để đề phòng mất điện hoặc sập hệ thống làm hỏng file lưu trữ.

---

## 8. Lộ trình triển khai Phase 3 (Session Roadmap)

### Phiên 3.1: Định nghĩa Gaussian & Khởi tạo (Initialization)
* Triển khai lớp lưu trữ thuộc tính Gaussian.
* Triển khai giải thuật tính khoảng cách 3-NN cho scale và ánh xạ màu RGB sang SH bậc 0.
* **Tiêu chí kiểm nghiệm**: Số Gaussian khởi tạo đúng bằng số điểm sparse có train support; các thuộc tính nằm trong dải giá trị vật lý quy định; phép biến đổi chuẩn hóa thế giới khớp với Phase 2.

### Phiên 3.2: Bộ điều phối render (gsplat Renderer Adapter)
* Xây dựng wrapper sạch kết nối `gsplat.rasterization` nhận camera và trả về `rgb`, `alpha`, `depth`.
* **Tiêu chí kiểm nghiệm**: Splat chiếu chính xác trên trục quang; dịch chuyển camera tạo dịch chuyển tương ứng trên ảnh; gradient đạo ngược hợp lệ trên toàn bộ tham số đầu vào.

### Phiên 3.3: Hàm Loss & Optimizer
* Triển khai Masked L1 và Masked SSIM sử dụng thuật toán xói mòn mask.
* Tạo cấu trúc optimizer groups và lịch trình giảm LR của tọa độ Mean.
* **Tiêu chí kiểm nghiệm**: Ảnh dự đoán trùng khớp hoàn toàn với ảnh thật sẽ cho loss bằng 0; thay đổi các pixel ngoài mask không gây biến động loss.

### Phiên 3.4: Tích hợp Density Control (Adaptive Strategy)
* Xây dựng adapter bao bọc `gsplat.DefaultStrategy` liên kết với quá trình forward và backward.
* **Tiêu chí kiểm nghiệm**: Pruning loại bỏ đúng các Gaussian mờ; duplicate và split phân tách các Gaussians có gradient lớn; số lượng Gaussian biến đổi chính xác theo lịch trình.

### Phiên 3.5: Vòng lặp huấn luyện & Checkpoint
* Viết script chạy vòng lặp tối ưu chính, ghi nhận log và chỉ số.
* Triển khai tính năng lưu/khôi phục checkpoint an toàn nguyên tử (atomic writing/RNG restore).
* **Tiêu chí kiểm nghiệm**: Huấn luyện thành công 1,000 steps trên scene nhỏ mà không gặp lỗi NaN/Inf hay tràn bộ nhớ (OOM); phục hồi checkpoint chạy tiếp tạo ra kết quả trùng khớp với chạy liên tục.

### Phiên 3.6: Smoke Test kiểm chứng hội tụ thực tế
* Thử nghiệm huấn luyện trên 1 scene thực tế (HCM0181) với hai cấu hình:
  - **Run A**: Thu nhỏ ảnh 4 lần, huấn luyện 500 steps kiểm tra độ bền hệ thống.
  - **Run B**: Thu nhỏ ảnh 2 lần, huấn luyện 7,000 steps kiểm tra chất lượng hội tụ thực tế.
* **Tiêu chí kiểm nghiệm**: Peak VRAM dưới 23GB; điểm số PSNR/SSIM tăng rõ rệt so với lúc khởi tạo; ảnh kết xuất không bị méo lệch hướng hoặc trống rỗng.
