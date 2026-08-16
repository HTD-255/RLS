# Hướng Dẫn Chạy Thực Nghiệm

## 📋 Cấu Trúc Project

```
c:\Research\Intercept\
├── controllers/              ← 4 controllers (LR, PID, RLS, MPC)
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
├── utils/                    ← Cấu hình, logging, metrics
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── metrics.py
│
├── analysis/                 ← Vẽ đồ thị, thống kê
│   ├── __init__.py
│   ├── plot_results.py
│   └── main.py
│
├── data/                     ← Kết quả thí nghiệm
│   ├── exp1/
│   ├── exp2/
│   ├── exp3/
│   └── exp4/
│
└── figures/                  ← Đồ thị xuất ra PDF
    ├── comparison_boxplot.pdf
    ├── lambda_sweep.pdf
    ├── perturbation_response.pdf
    ├── multisensor_comparison.pdf
    └── summary_table.txt
```

---

## 🚀 Cách Chạy Thực Nghiệm

### Cách 1: Chạy Từng Thí Nghiệm Riêng Lẻ

**Exp 1: Straight-Line Calibration (So sánh 4 phương pháp)**
```bash
cd c:\Research\Intercept
python experiments/exp1_straight_line.py
```
- Chạy 4 controllers (LR, PID, RLS, MPC) trên track thẳng
- Mỗi controller chạy 10 lần (tùy chỉnh `NUM_TRIALS_PER_CONDITION` trong `utils/config.py`)
- Kết quả: CSV + NPZ files trong `data/exp1/`
- Metrics: MAE_yaw, RMSE_yaw, convergence_time

**Exp 2: Lambda Sweep (Tìm λ tối ưu cho RLS)**
```bash
python experiments/exp2_lambda_sweep.py
```
- Test 6 giá trị λ: [0.90, 0.92, 0.95, 0.97, 0.99, 1.00]
- Mỗi λ chạy 10 lần
- Metrics: convergence_time, mae_yaw, theta_variance

**Exp 3: Perturbation Test (Test robustness)**
```bash
python experiments/exp3_perturbation.py
```
- Perturbation tại t=3s (thêm vật nặng)
- So sánh khả năng hồi phục của 4 controllers
- Metrics: recovery_time, peak_error, mae_yaw

**Exp 4: Multi-Sensor Comparison**
```bash
python experiments/exp4_multisensor.py
```
- RLS với 3 cấu hình cảm biến:
  - `gyro_only`: chỉ gyro_z
  - `euler_only`: chỉ euler_yaw
  - `full_imu`: gyro_z + euler_yaw + accel_y
- Metrics: convergence_time, mae_yaw, rmse_yaw

### Cách 2: Chạy Tất Cả 4 Thí Nghiệm

```bash
python analysis/main.py
```
- Chạy tuần tự: Exp 1 → Exp 2 → Exp 3 → Exp 4
- Tự động vẽ đồ thị offline
- Tạo bảng tóm tắt kết quả

---

## ⚙️ Tùy Chỉnh Cấu Hình

Sửa file `utils/config.py`:

```python
# Số lần chạy mỗi thí nghiệm
NUM_TRIALS_PER_CONDITION = 10  # Thay 10 thành số lần muốn chạy

# Thời gian thí nghiệm
EXP1_DURATION = 5.0    # 5 giây cho Exp 1
EXP3_DURATION = 8.0    # 8 giây cho Exp 3 (perturbation ở t=3s)

# Tham số RLS
RLS_DEFAULT_LAMBDA = 0.95      # Forgetting factor mặc định
RLS_LAMBDA_VALUES = [0.90, 0.92, 0.95, 0.97, 0.99, 1.00]  # Giá trị sweep Exp 2

# Tham số PID
PID_KP = 0.5
PID_KI = 0.1
PID_KD = 0.05
```

---

## 📊 Đầu Ra Từng Thí Nghiệm

### Exp 1 Output (`data/exp1/`)
```
exp1_LR_trial0.csv       ← Raw data (timestamp, gyro_z, euler_yaw, ...)
exp1_LR_trial0.npz       ← Numpy archive
...
exp1_RLS_trial9.npz      ← 10 trials mỗi controller
results.npz              ← Aggregated results
```

### Exp 2 Output (`data/exp2/`)
```
exp2_lambda0.90_trial0.csv
exp2_lambda0.90_trial0.npz
...
exp2_lambda1.00_trial9.npz
results.npz              ← All lambda sweep results
```

### Exp 3 Output (`data/exp3/`)
```
exp3_LR_trial0.csv       ← Includes perturbation event
exp3_RLS_trial0.npz
...
results.npz              ← Recovery metrics
```

### Exp 4 Output (`data/exp4/`)
```
exp4_gyro_only_trial0.npz
exp4_euler_only_trial0.npz
exp4_full_imu_trial0.npz
...
results.npz
```

---

## 📈 Đồ Thị Được Tạo

Chạy sau khi có kết quả:
```bash
python -c "from analysis.plot_results import run_full_analysis; run_full_analysis()"
```

**Figures tạo ra:**
- `comparison_boxplot.pdf` - So sánh 4 controllers (Exp 1 + Exp 3)
- `lambda_sweep.pdf` - Phân tích λ (Exp 2): convergence time, MAE, variance
- `perturbation_response.pdf` - Response sau perturbation (Exp 3)
- `multisensor_comparison.pdf` - Fusion vs single sensor (Exp 4)
- `summary_table.txt` - Bảng tóm tắt tất cả metrics

---

## 📖 Cách Sử Dụng Từng Module

### 1. Controllers

```python
from controllers import RLSController

# Khởi tạo RLS với λ = 0.95
rls = RLSController(n_features=4, forgetting_factor=0.95, delta=100.0)

# Mỗi timestep
sensor_data = {
    'gyro_z': 1.5,       # rad/s
    'euler_yaw': 0.2,    # rad
    'accel_y': 0.5,      # m/s²
    'target': 0.0        # steering target
}

steering_cmd = rls.compute(sensor_data)  # → float in [-1, 1]

# Lấy tham số
theta = rls.get_parameters()  # [θ₁, θ₂, θ₃, θ₀]
conv_time = rls.compute_convergence_time()
```

### 2. Data Logger

```python
from utils import DataLogger

logger = DataLogger('exp1_RLS', trial_id=0, output_dir='./data/exp1')

# Log từng sample
for t in range(100):
    sensor_data = get_sensors()
    cmd = controller.compute(sensor_data)
    
    logger.log_sample(
        timestamp=t * 0.1,
        gyro_z=sensor_data['gyro_z'],
        euler_yaw=sensor_data['euler_yaw'],
        accel_y=sensor_data['accel_y'],
        steering_cmd=cmd,
        error=cmd - reference_cmd
    )

# Log metrics
logger.log_metrics(
    mae_yaw=1.23,
    rmse_yaw=1.45,
    convergence_time=25
)

# Save
npz_file = logger.save()  # → Trả về path
```

### 3. Metrics Computation

```python
from utils import (
    compute_mae,
    compute_rmse,
    compute_recovery_time,
    statistical_test_wilcoxon,
    effect_size_cohens_d
)

# Compute metrics
mae = compute_mae(errors)
rmse = compute_rmse(errors)

# Recovery time (Exp 3)
recovery_steps = compute_recovery_time(
    errors,
    perturbation_idx=30,
    threshold=2.0  # 2 degrees
)

# Statistical test
rls_mae = np.array([1.23, 1.45, 1.32, ...])
pid_mae = np.array([2.10, 2.35, 2.18, ...])

stat, pvalue = statistical_test_wilcoxon(rls_mae, pid_mae)
d = effect_size_cohens_d(rls_mae, pid_mae)

print(f"p-value: {pvalue:.4f}")
print(f"Cohen's d: {d:.4f}")
```

---

## 🔍 Troubleshooting

**Q: Bước nào chậm nhất?**
- Exp 1, 3: Chạy 4 controllers × 10 trials = 40 lần simulation
- Exp 2: 6 lambdas × 10 trials = 60 lần simulation
- Exp 4: 3 configs × 10 trials = 30 lần simulation

→ Giải pháp: Sửa `NUM_TRIALS_PER_CONDITION = 2` để test nhanh

**Q: Kết quả thế nào là tốt?**
- RLS recovery time < PID recovery time (Exp 3)
- RLS MAE ≈ MPC MAE < PID MAE < LR MAE (Exp 1)
- Optimal λ ≈ 0.95 (tính từ Exp 2)

**Q: Làm sao import module không được?**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers import RLSController  # ← Bây giờ ok
```

---

## 📝 Ví Dụ Chạy Quick Test (2 trials)

```bash
# Quick test: 2 trials mỗi thí nghiệm (~2 phút tổng cộng)
cd c:\Research\Intercept

# Sửa config
python -c "
import sys
sys.path.insert(0, '.')
from utils import config
config.NUM_TRIALS_PER_CONDITION = 2
"

# Chạy tất cả
python analysis/main.py
```

---

## ✅ Checklist Thực Hiện

```
□ Đọc hiểu research_plan.md
□ Cài đặt dependencies: numpy, scipy, matplotlib
□ Sửa config.py (số trials, thời gian, λ values)
□ Chạy Exp 1: python experiments/exp1_straight_line.py
□ Chạy Exp 2: python experiments/exp2_lambda_sweep.py
□ Chạy Exp 3: python experiments/exp3_perturbation.py
□ Chạy Exp 4: python experiments/exp4_multisensor.py
□ Vẽ đồ thị: python -c "from analysis.plot_results import run_full_analysis; run_full_analysis()"
□ Xem kết quả trong figures/
□ Viết bài báo
```

---

**Câu hỏi? Xem `research_plan.md` để hiểu rõ lý thuyết!**
