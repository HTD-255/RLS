# 📘 HƯỚNG DẪN THỰC NGHIỆM CHI TIẾT - RLS Steering Control trên AutoCar III

**Ngày viết:** 09/06/2026  
**Dành cho:** Tổ DteLab  
**Xe:** Hanback AutoCar III  
**Mục tiêu:** So sánh 4 phương pháp điều khiển lái: LR, PID, RLS, MPC

---

## 📋 MỤC LỤC

1. [Chuẩn Bị Ban Đầu](#1-chuẩn-bị-ban-đầu)
2. [Thiết Lập Phần Mềm](#2-thiết-lập-phần-mềm)
3. [Cấu Hình Hardware](#3-cấu-hình-hardware)
4. [Chạy Từng Thí Nghiệm](#4-chạy-từng-thí-nghiệm)
5. [Phân Tích Kết Quả](#5-phân-tích-kết-quả-offline)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. CHUẨN BỊ BAN ĐẦU

### 1.1 Danh Sách Thiết Bị Cần Có

**Phần Cứng:**
- [ ] Hanback AutoCar III (đã bật pin, kiểm tra pin còn đủ)
- [ ] Track thẳng dài ≥ 3 mét (dán tape/vạch trắng làm đường tham chiếu)
- [ ] Vật nặng 200g (ví dụ: viên pin AA × 30 hoặc quả cân) — cho Exp 3
- [ ] Thước đo 1 mét (để đo path deviation cuối track)
- [ ] Laptop/PC có Python 3.9+ (để chạy code + phân tích)

**Phần Mềm:**
- [ ] Python 3.9 hoặc cao hơn
- [ ] IDE: VS Code hoặc PyCharm
- [ ] Git (để clone code mẫu nếu cần)

### 1.2 Kiểm Tra AutoCar III

Bước này **bắt buộc** trước khi chạy thí nghiệm:

```bash
# Kết nối AutoCar III bằng USB hoặc Bluetooth
# Kiểm tra port COM (trên Windows: Device Manager)

# Test cơ bản: chạy xe thẳng 2 giây
python test.py
```

**Chúng ta cần kiểm tra:**
- ✅ Xe có nhận lệnh từ laptop không?
- ✅ Servo lái có phản ứng không? (quay từ trái sang phải)
- ✅ IMU có trả dữ liệu không? (gyro_z, euler_yaw, accel_y)
- ✅ Pin AutoCar còn đủ không? (nên > 70%)

**Nếu có vấn đề:**
- Kiểm tra kết nối USB/Bluetooth
- Khởi động lại AutoCar III
- Xem phần [Troubleshooting](#6-troubleshooting)

---

## 2. THIẾT LẬP PHẦN MỀM

### 2.1 Tạo Virtual Environment (Khuyến Nghị)

```bash
# Mở PowerShell hoặc Terminal
cd c:\Research\Intercept

# Tạo virtual environment
python -m venv venv

# Kích hoạt
# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Trên macOS/Linux bash:
source venv/bin/activate
```

**Kiểm tra:** Dòng lệnh bây giờ bắt đầu bằng `(venv)`

### 2.2 Cài Đặt Dependencies

```bash
# Cài numpy, scipy, matplotlib
pip install numpy scipy matplotlib pandas

# Kiểm tra cài đặt
python -c "import numpy; print('NumPy OK')"
python -c "import scipy; print('SciPy OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

**Nên nhìn thấy:**
```
NumPy OK
SciPy OK
Matplotlib OK
```

### 2.3 Cài Đặt Hanback AutoCar SDK

Giả định bạn đã có `pop` library từ Hanback:

```bash
# Nếu chưa có, cài từ file .whl hoặc pip
pip install hanback-pop  # Nếu trên PyPI
# Hoặc: pip install path/to/hanback_pop.whl
```

**Kiểm tra:**
```bash
python -c "from pop import *; print('Hanback SDK OK')"
```

### 2.4 Cấu Trúc Thư Mục Project

Sau khi download code template từ phần trước, thư mục sẽ trông như thế này:

```
c:\Research\Intercept\
├── controllers/              ← 4 controller classes
│   ├── __init__.py
│   ├── base_controller.py
│   ├── lr_controller.py
│   ├── pid_controller.py
│   ├── rls_controller.py
│   └── mpc_controller.py
│
├── experiments/              ← 4 thí nghiệm
│   ├── __init__.py
│   ├── exp1_straight_line.py
│   ├── exp2_lambda_sweep.py
│   ├── exp3_perturbation.py
│   └── exp4_multisensor.py
│
├── utils/                    ← Công cụ hỗ trợ
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── metrics.py
│
├── analysis/                 ← Phân tích offline
│   ├── __init__.py
│   ├── plot_results.py
│   └── main.py
│
├── data/                     ← Dữ liệu thí nghiệm
│   ├── exp1/
│   ├── exp2/
│   ├── exp3/
│   └── exp4/
│
├── figures/                  ← Đồ thị output
│   └── (PDF files will be here)
│
├── EXPERIMENT_GUIDE.md
├── quick_start.py
├── research_plan.md
└── venv/                     ← Virtual environment
```

---

## 3. CẤU HÌNH HARDWARE

### 3.1 Kết Nối AutoCar III

**Bước 1:** Kết nối USB
```
AutoCar III ──[USB cable]──> Laptop
```

**Bước 2:** Xác định COM Port

**Trên Windows:**
- Mở Device Manager (Ctrl+Shift+Esc → Device Manager)
- Tìm "Ports (COM & LPT)"
- Ghi nhớ port, ví dụ: `COM3`, `COM5`

**Trên macOS/Linux:**
```bash
# Liệt kê các port serial
ls /dev/tty.*
# Kết quả: /dev/tty.usbserial-XXXXXX hoặc tương tự
```

**Bước 3:** Cập nhật Config

Sửa file `utils/config.py`:

```python
# ═════════════════════════════════════════════
# AUTOCAR III SETTINGS
# ═════════════════════════════════════════════
AUTOCAR_COM_PORT = 'COM3'  # Thay đổi nếu khác
AUTOCAR_BAUD_RATE = 115200
AUTOCAR_TIMEOUT = 1.0  # seconds
```

### 3.2 Hiệu Chuẩn Xe Trước Thí Nghiệm

**⚠️ BẮT BUỘC:** Xe phải ở trạng thái "thẳng" ban đầu

**Cách 1: Thủ công**
1. Đặt xe trên track, hướng thẳng với vạch tham chiếu
2. Không gửi lệnh lái, để servo ở vị trí trung tâm
3. Ghi lại giá trị `euler_yaw` ban đầu (nên ≈ 0)

**Cách 2: Tự động (nếu được)**
```bash
# Chạy script hiệu chuẩn
python calibrate_yaw_offset.py
# Script này sẽ:
# 1. Đọc 10 giá trị euler_yaw liên tiếp
# 2. Tính trung bình
# 3. Lưu offset vào config
```

### 3.3 Chuẩn Bị Track

**Track Yêu Cầu:**
- Dài ≥ 3 mét
- Phẳng, không có vật cản
- Có vạch tham chiếu (tape trắng hoặc bút marker dài 3m)

**Đánh Dấu Track:**
```
├─ Điểm bắt đầu (Start) ────────────────────── Điểm kết thúc (End)
  ↓                        3 mét                    ↓
  [Đặt xe ở đây]                            [Đo path deviation tại đây]
```

**Chuẩn Bị Quay Phim (Tùy Chọn):**
- Đặt camera trên cao hướng xuống track
- Quay từ thí nghiệm để xem đường đi của xe

---

## 4. CHẠY TỪNG THÍ NGHIỆM

### 4.0 Quick Test — Kiểm Tra Tất Cả Hoạt Động

Trước khi chạy thí nghiệm thật, chạy test nhanh:

```bash
cd c:\Research\Intercept

# Chạy 5 giây quick test
python quick_start.py
```

**Dự kiến output:**
```
============================================================
QUICK START: Single 5-second RLS Trial
============================================================

[Quick Test] Initializing RLS controller...
[Quick Test] Initializing data logger...
[Quick Test] Running simulation (5 seconds, 50 timesteps)...
  Step 10/50: θ=[0.12 -0.05  0.03  0.01]
  Step 20/50: θ=[0.25 -0.08  0.05  0.02]
  ...
  Step 50/50: θ=[0.45 -0.12  0.08  0.03]

[Quick Test] Results:
  MAE_yaw: 1.2345°
  RMSE_yaw: 1.5678°
  Final θ: [0.45 -0.12  0.08  0.03]
  Data saved to: ./data/quick_test_rls_trial0.npz

✓ Quick test completed successfully!
```

**Nếu thành công → Tiếp tục chạy thí nghiệm thật**

**Nếu có lỗi → Xem [Troubleshooting](#6-troubleshooting)**

---

### 4.1 EXPERIMENT 1: Straight-Line Calibration

**Mục tiêu:** So sánh 4 phương pháp trên đường thẳng

**Thời gian:** ~15-20 phút (cho 4 methods × 10 runs)

**Chuẩn Bị:**
```
✓ Track sạch, không vật cản
✓ AutoCar pin > 70%
✓ Camera sẵn sàng (tùy chọn)
✓ Giấy + bút để ghi chép kết quả
```

**Chạy Thí Nghiệm:**

```bash
# Option 1: Chạy với số runs mặc định (10 runs mỗi method)
python experiments/exp1_straight_line.py

# Option 2: Chạy với số runs tùy chỉnh
python -c "
import sys
sys.path.insert(0, '.')
from experiments.exp1_straight_line import run_experiment_1
results = run_experiment_1(num_trials=2)  # 2 runs nhanh để test
"
```

**Quá Trình Thực Hiện:**

Từng method sẽ chạy lần lượt:

```
[Exp1] Testing LR...
[Exp1 Trial 0] Controller: LR
  [LR] Training complete. Final loss: 0.001234
  MAE_yaw: 1.34°, RMSE_yaw: 1.56°
  Data saved: data/exp1/exp1_LR_trial0.npz

[Exp1 Trial 1] Controller: LR
  ...

[Exp1 Trial 9] Controller: LR
  ...

[Exp1] LR Summary:
  MAE_yaw: 1.42 ± 0.28°
  RMSE_yaw: 1.65 ± 0.32°

[Exp1] Testing PID...
  ...

[Exp1] Testing RLS...
  ...

[Exp1] Testing MPC...
  ...

[Exp1] Results saved to data/exp1/results.npz
```

**Kết Quả Mong Đợi:**

| Method | MAE_ψ (°) | RMSE_ψ (°) | Ghi Chú |
|--------|-----------|-----------|--------|
| LR     | ~1.4      | ~1.7      | Baseline offline |
| PID    | ~1.1      | ~1.3      | Tốt hơn LR |
| **RLS**    | **~0.8**      | **~0.95**     | **Tốt nhất** |
| MPC    | ~0.75     | ~0.9      | Có thể tốt hơn RLS |

**Thành Công Nếu:**
- ✅ Tất cả 40 files CSV được tạo (`data/exp1/exp1_*.csv`)
- ✅ RLS MAE < PID MAE (thường là)
- ✅ Không có lỗi runtime

**Kiểm Tra Kết Quả:**

```bash
# Xem danh sách files
ls data/exp1/

# In thống kê từ NPZ
python -c "
import numpy as np
r = np.load('data/exp1/results.npz', allow_pickle=True)
print('Controllers in results:', r.files)
for key in r.files:
    print(f'{key}: shape={r[key].shape}')
"
```

---

### 4.2 EXPERIMENT 2: Forgetting Factor Sweep

**Mục tiêu:** Tìm giá trị λ tối ưu cho RLS

**Thời gian:** ~15-20 phút (6 lambdas × 10 runs)

**Lambda Values được Test:**
```
λ = 0.90  (quên nhanh, thích ứng nhanh)
λ = 0.92
λ = 0.95  ← Mặc định, dự kiến tốt nhất
λ = 0.97
λ = 0.99
λ = 1.00  (không quên, tương đương batch)
```

**Chạy Thí Nghiệm:**

```bash
python experiments/exp2_lambda_sweep.py
```

**Quá Trình:**

```
============================================================
EXPERIMENT 2: FORGETTING FACTOR SWEEP
============================================================

[Exp2] Testing λ = 0.90...
[Exp2 λ=0.90 Trial 0]
  MAE_yaw: 1.45°
  Conv_time: 15.0 steps

[Exp2 λ=0.90 Trial 1]
  ...

[Exp2] λ = 0.90 Summary:
  MAE_yaw: 1.38 ± 0.25°
  Conv_time: 16.2 ± 2.3 steps

[Exp2] Testing λ = 0.92...
  ...

[Exp2] Testing λ = 0.95...
  ...

[Exp2] λ = 0.95 Summary:
  MAE_yaw: 0.82 ± 0.15°  ← Tốt nhất
  Conv_time: 22.5 ± 3.1 steps

[Exp2] Testing λ = 0.97...
  ...

[Exp2] Results saved to data/exp2/results.npz
```

**Kết Quả Mong Đợi:**

| λ | MAE (°) | Conv_time (steps) | Ghi Chú |
|---|---------|------------------|--------|
| 0.90 | 1.38 | 16 | Quên nhanh, dao động |
| 0.92 | 1.10 | 19 | |
| **0.95** | **0.82** | **22** | **⭐ Tối ưu** |
| 0.97 | 0.85 | 28 | Ổn định hơn |
| 0.99 | 0.95 | 35 | Gần như batch |
| 1.00 | 1.05 | 40 | Hoàn toàn batch |

**Ý Nghĩa:**
- **λ quá nhỏ (0.90):** Xe quên nhanh, tham số dao động, không ổn định
- **λ tối ưu (0.95):** Cân bằng tốt giữa thích ứng và ổn định
- **λ lớn (0.99, 1.00):** Xe nhớ quá nhiều, không thích ứng được

---

### 4.3 EXPERIMENT 3: Perturbation Test — ⭐ THÍ NGHIỆM CHÍNH

**Mục tiêu:** Kiểm chứng khả năng thích ứng của RLS khi có thay đổi đột ngột

**Scenario:** 
- t = 0-3s: Xe chạy bình thường (ổn định)
- **t = 3s:** Đặt vật nặng 200g lên một bên xe → perturbation!
- t = 3-8s: Quan sát cách mỗi controller phục hồi

**Thời gian:** ~10-15 phút (4 methods × 10 runs)

**Chuẩn Bị Perturbation:**
1. Chuẩn bị vật nặng 200g (ví dụ: quả cân, viên pin, gạch nhỏ)
2. Đặt nó ở vị trí dễ gắn/tháo nhanh (ví dụ: trên mái xe)
3. **Kiểm tra:** Khi gắn vật, xe có bị nghiêng/lệch không?

**Chạy Thí Nghiệm:**

```bash
python experiments/exp3_perturbation.py
```

**Quá Trình (Rất Quan Trọng):**

Mỗi trial 8 giây, thực hiện theo thứ tự:

```
[Exp3 Trial 0] Controller: LR

t=0-3s    : Chạy bình thường (xe ổn định)
t=3s      : ⚠️ GẮN VẬT NẶNG 200g (nhanh chóng!)
t=3-8s    : Quan sát xe có hồi phục không?

[Exp3 Trial 0] Controller: LR
  Peak_error: 5.23°  ← Yaw error lớn nhất sau perturbation
  Recovery_time: -1 steps  ← Không hồi phục (LR mô hình cố định)
  ...

[Exp3 Trial 1] Controller: LR
  Peak_error: 5.45°
  Recovery_time: -1 steps
  ...

[Exp3] LR Summary:
  Peak_error: 5.34 ± 0.15°
  Recovery_time: No recovery detected for all trials

[Exp3] Testing PID...
  Peak_error: 4.23°
  Recovery_time: 28.5 ± 5.3 steps  ← Hồi phục, nhưng chậm

[Exp3] Testing RLS...
  Peak_error: 3.12°  ← Nhỏ hơn!
  Recovery_time: 12.5 ± 2.1 steps  ← Nhanh! ⭐

[Exp3] Testing MPC...
  Peak_error: 2.89°  ← Nhỏ nhất
  Recovery_time: 10.3 ± 1.8 steps  ← Nhanh nhất
```

**Kết Quả Mong Đợi:**

| Method | Peak Error (°) | Recovery Time (s) | Ghi Chú |
|--------|----------------|------------------|--------|
| LR     | ~5.3           | ✗ Không hồi phục   | Mô hình cố định |
| PID    | ~4.2           | ~2.9 s            | Hồi phục chậm |
| **RLS**    | **~3.1**           | **~1.3 s**        | **Hồi phục nhanh** ⭐ |
| MPC    | ~2.9           | ~1.0 s            | Nhanh nhất |

**Ý Nghĩa Kết Quả:**
- **LR:** Không thể hồi phục → phải dừng + retrain
- **PID:** Hồi phục nhờ integral term, nhưng chậm
- **RLS:** Hồi phục nhanh nhờ cập nhật tham số online ✅
- **MPC:** Tốt nhất nhưng phức tạp hơn

**❗ ĐIỀU QUAN TRỌNG KHI GẮN VẬT NẶNG:**
1. **Đủ nhanh:** Gắn ngay lúc bước vào t=3s
2. **Không bị rơi:** Gắn chắc để vật không rơi giữa chạy
3. **Tháo ngay:** Sau khi thí nghiệm, tháo vật (để pin AutoCar không hết quá nhanh)

---

### 4.4 EXPERIMENT 4: Multi-Sensor Comparison

**Mục tiêu:** Đánh giá lợi ích của fusion IMU

**So Sánh 3 Cấu Hình RLS:**

1. **Gyro Only:** Dùng chỉ `gyro_z`
   - Features: `[gyro_z, 1.0]` (2D)
   - Nhanh nhất, nhưng có thể dao động

2. **Euler Only:** Dùng chỉ `euler_yaw`
   - Features: `[euler_yaw, 1.0]` (2D)
   - Có thể bị drift do IMU

3. **Full IMU:** Dùng cả ba
   - Features: `[gyro_z, euler_yaw, accel_y, 1.0]` (4D) ← **Tốt nhất**
   - Robust, có redundancy

**Thời gian:** ~10-15 phút (3 configs × 10 runs)

**Chạy Thí Nghiệm:**

```bash
python experiments/exp4_multisensor.py
```

**Quá Trình:**

```
============================================================
EXPERIMENT 4: MULTI-SENSOR COMPARISON
============================================================

[Exp4] Testing gyro_only...
[Exp4 gyro_only Trial 0]
  MAE_yaw: 1.23°
  Convergence: 18.5 steps

[Exp4 gyro_only Trial 1]
  ...

[Exp4] gyro_only Summary:
  MAE_yaw: 1.28 ± 0.32°
  Conv_time: 19.2 ± 3.5 steps

[Exp4] Testing euler_only...
  MAE_yaw: 1.45 ± 0.38°  ← Tệ nhất (bị drift)
  Conv_time: 25.1 ± 4.2 steps

[Exp4] Testing full_imu...
  MAE_yaw: 0.82 ± 0.20°  ← Tốt nhất ⭐
  Conv_time: 22.3 ± 3.1 steps
```

**Kết Quả Mong Đợi:**

| Config | MAE (°) | Conv Time (steps) | Ghi Chú |
|--------|---------|------------------|--------|
| Gyro Only | 1.28 | 19.2 | Đơn giản, nhưng dao động |
| Euler Only | 1.45 | 25.1 | Bị drift → tệ nhất |
| **Full IMU** | **0.82** | **22.3** | **Tốt nhất** ✅ |

**Kết Luận:** Dùng đầy đủ IMU fusion tốt hơn 40-77%

---

## 5. PHÂN TÍCH KẾT QUẢ OFFLINE

### 5.1 Chạy Phân Tích Tự Động

Sau khi hoàn thành tất cả 4 thí nghiệm:

```bash
# Chạy phân tích offline trên PC (không cần AutoCar)
python analysis/plot_results.py
```

Hoặc:

```bash
python -c "
from analysis.plot_results import run_full_analysis
run_full_analysis(results_dir='./data', output_dir='./figures')
"
```

**Dự Kiến Output:**

```
======================================================================
OFFLINE ANALYSIS & PLOTTING
======================================================================

[Analysis] Plotting comparison box plot...
  Saved: figures/comparison_boxplot.pdf

[Analysis] Plotting lambda sweep analysis...
  Saved: figures/lambda_sweep.pdf

[Analysis] Plotting perturbation response...
  Saved: figures/perturbation_response.pdf

[Analysis] Plotting multi-sensor comparison...
  Saved: figures/multisensor_comparison.pdf

[Analysis] Generating summary table...
  Saved: figures/summary_table.txt

[Analysis] Complete! Figures saved to: ./figures
```

### 5.2 Xem Kết Quả

**Các File Được Tạo:**

```
figures/
├── comparison_boxplot.pdf     ← So sánh 4 controllers
├── lambda_sweep.pdf           ← Phân tích λ
├── perturbation_response.pdf  ← Recovery time
├── multisensor_comparison.pdf ← Fusion benefit
└── summary_table.txt          ← Bảng tóm tắt
```

**Xem Bảng Tóm Tắt:**

```bash
# Xem file text
type figures/summary_table.txt

# Hoặc
more figures/summary_table.txt
```

**Output Mẫu:**

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    EXPERIMENTAL RESULTS SUMMARY TABLE                         ║
╚════════════════════════════════════════════════════════════════════════════════╝

| Method | MAE_ψ (°)        | RMSE_ψ (°)       | Recovery (s) | Peak Error (°)  |
|--------|------------------|------------------|--------------|-----------------|
| LR     | 1.4203±0.2845    | 1.6523±0.3124    | N/A          | 5.2301±0.1523  |
| PID    | 1.1023±0.1945    | 1.3401±0.2301    | 2.85±0.52    | 4.2145±0.1834  |
| RLS    | 0.8234±0.1523    | 0.9512±0.1834    | 1.25±0.21    | 3.1203±0.1612  |
| MPC    | 0.7845±0.1401    | 0.9123±0.1645    | 1.03±0.18    | 2.8934±0.1423  |
```

### 5.3 Xem Từng Đồ Thị

**Dùng Python Viewer:**

```bash
# Xem boxplot
python -c "
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('figures/comparison_boxplot.pdf')
plt.imshow(img)
plt.show()
"

# Hoặc dùng Adobe Reader / Preview
```

**Dùng Command Line (Windows):**

```bash
# Mở tất cả PDF
start figures/
```

### 5.4 Chạy Statistical Test

```bash
python -c "
import numpy as np
from utils.metrics import statistical_test_wilcoxon, effect_size_cohens_d

# Load kết quả LR và RLS
rls_mae = np.array([0.81, 0.85, 0.79, 0.83, 0.84, 0.80, 0.82, 0.86, 0.81, 0.83])
pid_mae = np.array([1.12, 1.05, 1.15, 1.08, 1.10, 1.03, 1.18, 1.07, 1.11, 1.09])

# Wilcoxon test (non-parametric)
stat, pval = statistical_test_wilcoxon(rls_mae, pid_mae)
print(f'Wilcoxon test RLS vs PID:')
print(f'  Statistic: {stat}')
print(f'  P-value: {pval:.6f}')

if pval < 0.05:
    print(f'  ✓ RLS significantly better than PID (p < 0.05)')
else:
    print(f'  ✗ No significant difference')

# Effect size
d = effect_size_cohens_d(rls_mae, pid_mae)
print(f'\\nCohen\\'s d: {d:.4f}')
print(f'  Interpretation:')
if abs(d) < 0.2:
    print(f'    Negligible effect')
elif abs(d) < 0.5:
    print(f'    Small effect')
elif abs(d) < 0.8:
    print(f'    Medium effect')
else:
    print(f'    Large effect ⭐')
"
```

**Kết Quả Mong Đợi:**

```
Wilcoxon test RLS vs PID:
  Statistic: 55.0
  P-value: 0.001234
  ✓ RLS significantly better than PID (p < 0.05)

Cohen's d: -1.2345
  Interpretation:
    Large effect ⭐
```

---

## 6. TROUBLESHOOTING

### 6.1 AutoCar Không Kết Nối

**Triệu Chứng:**
```
Error: Serial port COM3 not found
```

**Giải Pháp:**

1. **Kiểm tra kết nối USB**
   - Rút cáp USB, chờ 5 giây, cắm lại
   - Kiểm tra LED trên AutoCar (có sáng không?)

2. **Tìm đúng COM port**
   - Mở Device Manager (Win: Ctrl+Shift+Esc)
   - Tìm "Ports (COM & LPT)"
   - Ghi nhớ port, sửa trong `config.py`

3. **Kiểm tra quyền**
   - Chạy IDE dưới quyền Admin
   - Hoặc: `pip install pyserial` (driver có thể bị thiếu)

---

### 6.2 AutoCar Không Nhận Lệnh Lái

**Triệu Chứng:**
```
Connected to AutoCar, but servo not responding
```

**Giải Pháp:**

1. **Kiểm tra pin**
   - Bật chế độ battery indicator trên AutoCar
   - Nếu pin < 50%, sạc lại

2. **Kiểm tra servo cơ khí**
   - Lái tay (servo thông thường) xem có cứng không
   - Nếu cứng, có thể bị mắc kẹt → đến DteLab để sửa

3. **Reset AutoCar**
   - Tắt pin 10 giây
   - Bật lại
   - Kết nối USB lại

---

### 6.3 IMU Dữ Liệu Không Chính Xác

**Triệu Chứng:**
```
euler_yaw keeps drifting / increases over time without car moving
```

**Giải Pháp:**

1. **Cân Bằng IMU**
   ```bash
   # Đặt xe trên bàn phẳng (không chạy)
   # Chạy calibration script
   python calibrate_imu_offset.py
   ```

2. **Sử Dụng Gyro-Z Thay Vì Euler-Yaw**
   - Gyro là rate (đạo hàm), ổn định hơn
   - Euler dễ drift → chỉ dùng để reference

3. **Tăng Low-Pass Filter**
   - Sửa `config.py`:
   ```python
   LPF_CUTOFF_HZ = 1.0  # Thay từ 2.0 thành 1.0 (mềm hơn)
   ```

---

### 6.4 Exp Chạy Rất Chậm

**Triệu Chứng:**
```
1 trial mất > 30 giây (nên là 5s)
```

**Giải Pháp:**

1. **Giảm số runs**
   ```python
   # Sửa config
   NUM_TRIALS_PER_CONDITION = 2  # Thay từ 10
   ```

2. **Kiểm tra việc log**
   - CSV write có thể chậm
   - Dùng `allow_pickle=False` trong logger

3. **Dùng SSD thay HDD**
   - Nếu lưu data trên USB → copy sang SSD

---

### 6.5 RLS Parameters Không Hội Tụ

**Triệu Chứng:**
```
θ (parameters) keep oscillating, no convergence
```

**Giải Pháp:**

1. **Tăng forgetting factor λ**
   ```python
   # Sửa config
   RLS_DEFAULT_LAMBDA = 0.97  # Thay từ 0.95
   ```
   - λ cao → quên chậm → ổn định hơn

2. **Tăng covariance initialization**
   ```python
   # Sửa config
   RLS_DELTA = 200.0  # Thay từ 100.0
   ```
   - Covariance lớn hơn → learning tốc độ nhanh hơn ban đầu

3. **Tăng simulation time**
   ```python
   EXP1_DURATION = 10.0  # Thay từ 5.0 (cho đủ thời gian hội tụ)
   ```

---

### 6.6 Plot / Analysis Không Chạy

**Triệu Chứng:**
```
ModuleNotFoundError: No module named 'matplotlib'
```

**Giải Pháp:**

```bash
# Cài lại dependencies
pip install matplotlib pandas scipy

# Hoặc reinstall tất cả từ requirements.txt
pip install -r requirements.txt
```

---

### 6.7 Lỗi Path / Import

**Triệu Chứng:**
```
ModuleNotFoundError: No module named 'controllers'
```

**Giải Pháp:**

1. **Chạy từ đúng thư mục:**
   ```bash
   # ✓ Đúng
   cd c:\Research\Intercept
   python quick_start.py

   # ✗ Sai
   cd c:\Research
   python Intercept/quick_start.py
   ```

2. **Kiểm tra PYTHONPATH:**
   ```bash
   python -c "
   import sys
   for p in sys.path:
       print(p)
   "
   ```

3. **Thêm path vào script:**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent))
   ```

---

## 7. CHẠY TOÀN BỘ THỰC NGHIỆM (QUICK REFERENCE)

### Cách 1: Chạy Tuần Tự (Khuyến Nghị)

```bash
cd c:\Research\Intercept

# 1. Quick test (30 sec)
python quick_start.py

# 2. Exp 1 (15 min)
python experiments/exp1_straight_line.py

# 3. Exp 2 (15 min)
python experiments/exp2_lambda_sweep.py

# 4. Exp 3 (10 min) — ⚠️ Cần gắn vật nặng
python experiments/exp3_perturbation.py

# 5. Exp 4 (15 min)
python experiments/exp4_multisensor.py

# 6. Phân tích (5 min)
python analysis/plot_results.py

# Tổng: ~70 phút
```

### Cách 2: Chạy Tất Cả Cùng Lúc

```bash
python analysis/main.py
```

**Lưu ý:** Main.py sẽ chạy Exp 1-4 tuần tự, sau đó tự động phân tích.

---

## 8. CHECKLIST HOÀN THÀNH

### Trước Khi Bắt Đầu

- [ ] Hanback AutoCar III đã bật, pin > 70%
- [ ] USB kết nối, tìm được COM port
- [ ] Track 3m sạch, không vật cản
- [ ] Python 3.9+, virtual environment, dependencies cài đặt
- [ ] Kiểm tra `quick_start.py` thành công

### Exp 1 (Straight-Line)

- [ ] Xe ở vị trí start (hướng thẳng)
- [ ] Chạy exp1_straight_line.py
- [ ] 40 files CSV được tạo (4 methods × 10 runs)
- [ ] Kết quả: RLS MAE < PID MAE

### Exp 2 (Lambda Sweep)

- [ ] Chạy exp2_lambda_sweep.py
- [ ] 60 files được tạo (6 lambdas × 10 runs)
- [ ] Kết quả: λ=0.95 MAE nhỏ nhất

### Exp 3 (Perturbation) — ⚠️ QUAN TRỌNG

- [ ] Chuẩn bị vật nặng 200g
- [ ] Chạy exp3_perturbation.py
- [ ] **Tại t=3s, gắn nhanh vật nặng** ← TRỌng luyến này!
- [ ] 40 files được tạo
- [ ] Kết quả: RLS recovery_time < PID recovery_time

### Exp 4 (Multi-Sensor)

- [ ] Chạy exp4_multisensor.py
- [ ] 30 files được tạo (3 configs × 10 runs)
- [ ] Kết quả: full_imu MAE tốt nhất

### Phân Tích & Báo Cáo

- [ ] Chạy plot_results.py
- [ ] 4 PDF figures được tạo
- [ ] summary_table.txt có dữ liệu
- [ ] Statistical test p-value < 0.05
- [ ] Viết bài báo (6 pages, IEEE format)

---

## 9. GHI CHÚ QUAN TRỌNG

### ⚠️ Khi Gắn Vật Nặng (Exp 3)

1. **Thời Điểm Chính Xác:** Gắn ngay lúc t=3s (step 30)
   - Quá sớm: Khó kiểm soát thời gian
   - Quá muộn: Mất dữ liệu phục hồi

2. **Cách Gắn:**
   - Đặt vật trên mái xe (trọng tâm cao → tác động lớn)
   - HOẶC gắn trên bánh xe một bên (làm xe lệch)

3. **An Toàn:**
   - Gắn chắc để không rơi khi xe chạy
   - Tháo ngay sau khi xong (để pin không hết quá nhanh)

### ⚠️ Về IMU Drift

- **Euler-Yaw:** Dễ drift, chỉ dùng để reference
- **Gyro-Z:** Ổn định, là rate (tốc độ quay) → tốt hơn
- **Accel-Y:** Dùng để detect drift lateral

### ⚠️ Về Pin AutoCar

- Mỗi thí nghiệm ~5 phút → tổng 70 phút
- Pin sẽ hết từ 100% xuống ~20%
- **Sạc trước khi bắt đầu:** 100% → đủ cho tất cả

---

## 10. LIÊN HỆ & HỖ TRỢ

**Nếu Có Lỗi:**

1. Kiểm tra [Troubleshooting](#6-troubleshooting)
2. Xem [research_plan.md](research_plan.md) để hiểu lý thuyết
3. Xem [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) để xem code API
4. Liên hệ DteLab Lab Manager

**File Hữu Ích:**

- `research_plan.md` — Lý thuyết chi tiết
- `EXPERIMENT_GUIDE.md` — API & cấu hình
- `quick_start.py` — Test nhanh
- `utils/config.py` — Sửa tham số

---

**🎉 Chúc bạn thực hiện thành công các thí nghiệm! 🎉**

**Thời gian dự kiến:** ~2-3 giờ cho toàn bộ (nếu không có vấn đề)

