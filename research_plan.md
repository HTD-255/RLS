# Research Plan
## Online Self-Calibrating Steering Control Using Recursive Least Squares with IMU Feedback on a Small-Scale Autonomous Vehicle

> **Platform:** Hanback AutoCar III  
> **Target:** International Conference Paper (RIVF / ICCAIS / ATC)  
> **Timeline:** 6 tuần (Tháng 6–7/2026)  
> **Created:** DteLab

---

## 1. Bối Cảnh & Động Lực

### 1.1 Vấn đề thực tiễn

Xe tự hành quy mô nhỏ (small-scale autonomous vehicles) dùng trong giáo dục và nghiên cứu
thường gặp hiện tượng **lệch lái** (steering misalignment) do:

- Hao mòn cơ khí theo thời gian (bánh xe, servo)
- Thay đổi mặt đường (trơn, gồ ghề, nghiêng)
- Thay đổi tải trọng (gắn thêm thiết bị, pin yếu)
- Sai số lắp đặt ban đầu

### 1.2 Hạn chế của phương pháp hiện tại

Code mẫu của Hanback AutoCar III (notebook 8.1 & 8.2) sử dụng **Linear Regression offline**:

```
Thu thập dữ liệu → Train LR (5000 epochs) → Áp dụng cố định
```

**Nhược điểm:**
- ❌ Phải dừng xe để hiệu chuẩn lại khi điều kiện thay đổi
- ❌ Không thích ứng real-time
- ❌ Chỉ dùng 1 kênh sensor (Euler-yaw HOẶC Gyro-Z), bỏ phí IMU 9-DOF
- ❌ Mô hình tuyến tính cố định, không capture được phi tuyến

### 1.3 Đề xuất

Xây dựng **bộ điều khiển tự hiệu chuẩn online** dựa trên **Recursive Least Squares (RLS)**
với forgetting factor, sử dụng phản hồi đa kênh từ IMU, có khả năng:

- ✅ Tự thích ứng khi điều kiện vận hành thay đổi
- ✅ Không cần dừng xe để hiệu chuẩn lại
- ✅ Khai thác đa kênh IMU (Gyro-Z + Euler-yaw + Accel-Y)
- ✅ Chạy nhẹ trên embedded platform (Raspberry Pi / Jetson Nano)

---

## 2. Câu Hỏi Nghiên Cứu

| # | Câu hỏi | Loại |
|---|---------|------|
| **RQ1** | RLS online có cải thiện yaw error và path deviation so với LR offline trên cùng nền tảng AutoCar III không? | So sánh |
| **RQ2** | Forgetting factor λ ảnh hưởng thế nào đến trade-off giữa tốc độ thích ứng và ổn định? | Phân tích tham số |
| **RQ3** | Khi điều kiện vận hành thay đổi đột ngột (perturbation), RLS hội tụ lại nhanh hơn bao nhiêu so với các phương pháp khác? | Robustness |

---

## 3. Các Phương Pháp So Sánh

### 3.1 Linear Regression — Baseline (Offline)

**Nguồn:** Notebook 8.1 & 8.2 gốc của Hanback

```python
# Offline: thu thập → train → deploy cố định
LR = AI.Linear_Regression()
LR.X_data = dataset['gyro']   # hoặc euler
LR.Y_data = dataset['steer']
LR.train(times=5000, print_every=100)

# Online: dùng mô hình cố định
while driving:
    err = Car.getGyro('z')
    steer = LR.run([err])[0][0]
    Car.steering = steer * 1.5
```

**Đặc điểm:**
- Batch learning, offline only
- Mô hình: `steer = w₁ × sensor + w₀`
- Không thích ứng sau khi deploy

---

### 3.2 PID Controller (Online, model-free)

**Vai trò:** Baseline online đơn giản nhất, không cần mô hình

```python
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.1):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return np.clip(output, -1.0, 1.0)
```

**Đặc điểm:**
- Online, reactive
- Không học mô hình steering-yaw
- Cần tune thủ công Kp, Ki, Kd
- Không tự thích ứng khi hệ thống thay đổi

**Phương pháp tune:** Ziegler-Nichols hoặc manual sweep trên AutoCar III

---

### 3.3 Recursive Least Squares (RLS) — ★ CONTRIBUTION CHÍNH

**Vai trò:** Online self-calibration, tự học quan hệ steering-yaw liên tục

```python
class RLSCalibrator:
    """
    Online self-calibrating steering model using RLS.
    Model: steer = θ₁ × gyro_z + θ₂ × euler_yaw + θ₃ × accel_y + θ₀
    """
    def __init__(self, n_features=4, forgetting_factor=0.95, delta=100.0):
        self.λ = forgetting_factor
        self.θ = np.zeros(n_features)           # parameter vector
        self.P = np.eye(n_features) * delta      # covariance matrix

    def update(self, x, y_true):
        """
        x: feature vector [gyro_z, euler_yaw, accel_y, 1.0]
        y_true: actual steering correction needed
        """
        # Prediction error
        y_pred = x @ self.θ
        e = y_true - y_pred

        # Kalman gain
        Px = self.P @ x
        K = Px / (self.λ + x @ Px)

        # Update parameters
        self.θ += K * e

        # Update covariance
        self.P = (self.P - np.outer(K, x @ self.P)) / self.λ

        return y_pred, e

    def predict(self, x):
        return x @ self.θ
```

**Đặc điểm:**
- Online, recursive — cập nhật mỗi timestep
- Forgetting factor λ: quên dần dữ liệu cũ → thích ứng thay đổi
- Multi-sensor input: tận dụng Gyro-Z + Euler-yaw + Accel-Y
- Complexity: O(n²) per step với n = số features (rất nhẹ)

**Forgetting factor λ sweep:**

| λ | Ý nghĩa | Dự kiến |
|---|---------|---------|
| 0.90 | Quên rất nhanh | Thích ứng nhanh, dao động nhiều |
| 0.92 | Quên nhanh | Trade-off |
| **0.95** | **Cân bằng** | **Mặc định, dự kiến tốt nhất** |
| 0.97 | Quên chậm | Ổn định hơn, thích ứng chậm |
| 0.99 | Gần như nhớ hết | Gần giống batch LR |
| 1.00 | Nhớ tất cả | Tương đương OLS tích lũy |

---

### 3.4 Lightweight MPC (Online, model-based)

**Vai trò:** Upper bound về hiệu năng, so sánh với RLS

```python
class LightweightMPC:
    """
    Single-track (bicycle) model MPC with horizon N.
    State: [yaw, yaw_rate]
    Input: steering_cmd
    """
    def __init__(self, N=5, dt=0.1, Q_yaw=10.0, Q_rate=1.0, R_steer=0.1):
        self.N = N
        self.dt = dt
        self.Q = np.diag([Q_yaw, Q_rate])
        self.R = np.array([[R_steer]])

    def solve(self, current_state, target_yaw=0.0):
        """
        Solve QP: min Σ (x-xref)ᵀQ(x-xref) + uᵀRu
        s.t. x_{k+1} = Ax_k + Bu_k
             |u| <= 1.0
        """
        # Linearized bicycle model matrices
        # A, B estimated from current RLS parameters or pre-identified
        # ... QP solve using cvxpy or scipy.optimize ...
        pass
```

**Đặc điểm:**
- Predictive, optimal trong horizon
- Nặng hơn RLS, có thể chậm trên embedded
- Cần mô hình động lực xe (ước lượng hoặc dùng kết quả RLS)
- Handle constraints tường minh (|steering| ≤ 1)

---

## 4. Kiến Trúc Hệ Thống

```
╔══════════════════════════════════════════════════════════════╗
║                    AutoCar III Hardware                       ║
║  ┌──────────┐   ┌────────────────┐   ┌──────────┐           ║
║  │ Steering │   │   IMU 9-DOF    │   │ Motor /  │           ║
║  │  Servo   │   │ • Gyro (x,y,z) │   │ Encoder  │           ║
║  │ [-1, +1] │   │ • Accel(x,y,z) │   │ (speed)  │           ║
║  │          │   │ • Euler(y,p,r)  │   │          │           ║
║  └────┬─────┘   └───────┬────────┘   └────┬─────┘           ║
║       │                 │                  │                  ║
╚═══════╪═════════════════╪══════════════════╪══════════════════╝
        │                 │                  │
        │    ┌────────────┴────────────┐     │
        │    │   Sensor Preprocessing   │     │
        │    │ • Euler wrap-around fix  │     │
        │    │ • Low-pass filter        │     │
        │    │ • Timestamp sync         │     │
        │    └────────────┬────────────┘     │
        │                 │                  │
        │    ┌────────────┴────────────┐     │
        │    │   Feature Vector Build   │     │
        │    │ x = [gyro_z,             │     │
        │    │      euler_yaw,          │     │
        │    │      accel_y,            │     │
        │    │      1.0 (bias)]         │     │
        │    └────────────┬────────────┘     │
        │                 │                  │
        │         ┌───────┴───────┐          │
        │         ▼               ▼          │
        │   ┌───────────┐  ┌───────────┐     │
        │   │ Calibrator │  │  Data     │     │
        │   │ (LR/PID/   │  │  Logger   │     │
        │   │  RLS/MPC)  │  │ (CSV)     │     │
        │   └─────┬─────┘  └───────────┘     │
        │         │                           │
        │         ▼                           │
        │   ┌───────────────┐                 │
        │   │ Steering Cmd  │                 │
        │   │ clip(-1, +1)  │                 │
        │   └───────┬───────┘                 │
        │           │                         │
        ◄───────────┘                         │
                                              │
   ┌──────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│      Offline Analysis (PC)        │
│ • Yaw error time series           │
│ • Path deviation computation      │
│ • Convergence curve plotting      │
│ • Statistical tests (t-test)      │
│ • Forgetting factor sweep plot    │
└──────────────────────────────────┘
```

---

## 5. Thiết Kế Thực Nghiệm

### 5.1 Môi trường thí nghiệm

| Yếu tố | Thiết lập |
|---------|-----------|
| **Track** | Đường thẳng 3m, có vạch tham chiếu dán trên sàn |
| **Mặt sàn** | Gạch phẳng (condition A), thêm tấm foam (condition B) |
| **Tốc độ** | Cố định `Car.setSpeed(30)` — đủ chậm để đo chính xác |
| **Tần suất lấy mẫu** | 10 Hz (dt = 0.1s), khớp với code mẫu gốc |
| **Số lần lặp** | Mỗi thí nghiệm chạy **10 lần** → mean ± std |

### 5.2 Protocol thí nghiệm

#### Experiment 1: Straight-Line Calibration (Trả lời RQ1)

```
Mục tiêu: Xe chạy thẳng, đo khả năng bù lệch lái
Protocol:
  1. Đặt xe ở đầu track, hướng thẳng
  2. Khởi động controller (LR / PID / RLS / MPC)
  3. forward(30) trong 5 giây
  4. Ghi log: timestamp, gyro_z, euler_yaw, accel_y, steering_cmd
  5. Đo path deviation cuối track (thước / camera)
  6. Lặp lại 10 lần cho mỗi controller

Metrics:
  • Mean Absolute Yaw Error (°) = mean(|gyro_z|) during run
  • Final Path Deviation (cm) = khoảng cách cuối so với đường tham chiếu
  • Steady-State Error (°) = mean yaw error trong 2 giây cuối
```

#### Experiment 2: Forgetting Factor Sweep (Trả lời RQ2)

```
Mục tiêu: Tìm λ tối ưu cho RLS
Protocol:
  1. Cố định track thẳng, cùng điều kiện
  2. Chạy RLS với λ ∈ {0.90, 0.92, 0.95, 0.97, 0.99, 1.00}
  3. Mỗi λ chạy 10 lần
  4. Ghi log: θ convergence, yaw error, steering variance

Metrics:
  • Convergence Time (s) = thời gian θ ổn định (< 5% thay đổi)
  • Tracking Accuracy = mean yaw error sau convergence
  • Parameter Variance = var(θ) trong steady-state
  
Output: Plot λ vs (convergence_time, yaw_error, θ_variance)
```

#### Experiment 3: Perturbation Test (Trả lời RQ3) ★

```
Mục tiêu: Chứng minh adaptive advantage của RLS
Protocol:
  1. Chạy xe 3 giây (ổn định)
  2. Tại t=3s: ĐẶT THÊM VẬT NẶNG 200g lên một bên xe
     → Tạo perturbation, xe bị lệch
  3. Tiếp tục chạy thêm 5 giây
  4. Đo thời gian controller hội tụ lại
  5. So sánh: LR (phải retrain) vs PID vs RLS vs MPC

Metrics:
  • Recovery Time (s) = thời gian yaw error giảm về < 2° sau perturbation
  • Peak Yaw Error (°) = yaw error lớn nhất ngay sau perturbation
  • Steady-State Error After (°) = error ổn định sau perturbation

Kỳ vọng:
  • LR: KHÔNG hồi phục (mô hình cố định) → phải dừng + retrain
  • PID: hồi phục nhờ integral term, nhưng chậm
  • RLS: hồi phục nhanh nhờ online update
  • MPC: hồi phục nếu model update kịp
```

#### Experiment 4: Multi-Sensor Comparison (Bonus)

```
Mục tiêu: Đánh giá lợi ích multi-sensor fusion
Protocol:
  1. Chạy RLS (λ=0.95) với 3 cấu hình input:
     a) [gyro_z, 1]              — chỉ Gyro
     b) [euler_yaw, 1]           — chỉ Euler
     c) [gyro_z, euler_yaw, accel_y, 1]  — Full IMU
  2. Track thẳng, 10 lần mỗi cấu hình

Metrics: yaw_error, path_deviation, convergence_time
Kỳ vọng: (c) > (a) > (b) về tổng thể
```

---

## 6. Metrics & Đánh Giá

### 6.1 Bảng metrics chính

| Metric | Ký hiệu | Đơn vị | Cách tính |
|--------|----------|--------|-----------|
| Mean Absolute Yaw Error | MAE_ψ | ° (degree) | `mean(abs(gyro_z))` over run |
| Root Mean Square Yaw Error | RMSE_ψ | ° | `sqrt(mean(gyro_z²))` |
| Final Path Deviation | Δd | cm | Đo thước tại cuối track |
| Convergence Time | T_conv | s | Thời gian `abs(Δθ/θ) < 0.05` |
| Recovery Time | T_rec | s | Thời gian error < 2° sau perturbation |
| Computation Time | T_comp | ms/step | `time.perf_counter()` per loop |
| Parameter Stability | σ_θ | — | `std(θ)` trong steady-state |

### 6.2 Statistical Testing

- **Wilcoxon signed-rank test** (non-parametric, n=10 per group)
- Hoặc **paired t-test** nếu data đủ normal (Shapiro-Wilk check)
- Report **p-value** và **effect size** (Cohen's d)

---

## 7. Cấu Trúc Thư Mục Project

```
AutocarIII_Project/
├── research_plan.md                ← File này
├── samplecode/                     ← Code mẫu gốc Hanback (giữ nguyên)
│   ├── 8.1 휠 얼라인먼트.ipynb
│   ├── 8.2 차량 자세 제어.ipynb
│   └── ...
│
├── src/                            ← Source code thực nghiệm
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── base_controller.py      ← Abstract base class
│   │   ├── lr_controller.py        ← Linear Regression (baseline)
│   │   ├── pid_controller.py       ← PID Controller
│   │   ├── rls_controller.py       ← RLS Calibrator ★
│   │   └── mpc_controller.py       ← Lightweight MPC
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── collector.py            ← Automated data collection
│   │   └── sensor_preprocessor.py  ← Euler wrap-around, filtering
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── exp1_straight_line.py   ← Experiment 1
│   │   ├── exp2_lambda_sweep.py    ← Experiment 2
│   │   ├── exp3_perturbation.py    ← Experiment 3
│   │   └── exp4_multisensor.py     ← Experiment 4
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               ← CSV/JSON data logger
│       ├── metrics.py              ← Metric computation
│       └── config.py               ← Hyperparameters & constants
│
├── analysis/                       ← Offline analysis (chạy trên PC)
│   ├── plot_results.py
│   ├── statistical_tests.py
│   └── generate_tables.py
│
├── data/                           ← Raw experiment data
│   ├── exp1/
│   ├── exp2/
│   ├── exp3/
│   └── exp4/
│
├── figures/                        ← Figures cho paper
│   ├── system_architecture.pdf
│   ├── convergence_curves.pdf
│   ├── lambda_sweep.pdf
│   ├── perturbation_response.pdf
│   └── boxplot_comparison.pdf
│
├── paper/                          ← Bản thảo bài báo
│   ├── main.tex
│   ├── references.bib
│   └── figures/                    ← Symlink hoặc copy từ figures/
│
└── requirements.txt
```

---

## 8. Dự Kiến Figures Cho Paper

### Figure 1: System Architecture
- Block diagram hệ thống (vẽ từ Section 4)
- Gồm: Hardware → Preprocessing → Controller → Actuator

### Figure 2: Convergence Curves
- Plot θ (parameters) vs time cho RLS
- So sánh λ = {0.90, 0.95, 0.99}
- X: time (s), Y: θ values

### Figure 3: Yaw Error Time Series
- 4 subplot (LR, PID, RLS, MPC)
- X: time, Y: yaw error (°)
- Shaded region = std across 10 runs

### Figure 4: Perturbation Response ★
- Tất cả 4 controllers trên 1 plot
- Vertical dashed line tại t=3s (perturbation)
- Highlight recovery time cho mỗi controller

### Figure 5: Forgetting Factor Analysis
- Dual Y-axis: convergence time (left) vs yaw error (right)
- X: λ values
- Đánh dấu optimal λ

### Figure 6: Box Plot Comparison
- Box plot: MAE_ψ cho 4 controllers
- Có significance markers (*, **, ***)

### Table 1: Quantitative Results Summary
| Method | MAE_ψ (°) | RMSE_ψ (°) | Δd (cm) | T_conv (s) | T_rec (s) | T_comp (ms) |
|--------|-----------|-------------|---------|------------|-----------|-------------|
| LR     | — | — | — | N/A | N/A | — |
| PID    | — | — | — | N/A | — | — |
| RLS    | — | — | — | — | — | — |
| MPC    | — | — | — | — | — | — |

---

## 9. Dự Kiến Cấu Trúc Paper

```
Title: Online Self-Calibrating Steering Control Using Recursive Least
       Squares with IMU Feedback on a Small-Scale Autonomous Vehicle

Abstract (150-200 words)

I.   Introduction
     - Motivation: steering misalignment problem
     - Limitation of offline calibration
     - Contribution summary (3 bullets)

II.  Related Work
     - IMU-based vehicle calibration
     - Adaptive steering control
     - RLS in automotive applications
     - Gap: online self-calibration on educational platforms

III. System Description
     - AutoCar III hardware overview
     - IMU sensor specifications
     - Existing calibration approach (baseline)

IV.  Proposed Approach
     A. Problem Formulation
     B. RLS with Forgetting Factor
     C. Multi-Sensor Feature Vector
     D. Online Calibration Loop

V.   Experimental Setup
     - Test environment & protocol
     - Baseline methods (LR, PID, MPC)
     - Evaluation metrics

VI.  Results and Discussion
     A. Straight-Line Calibration (Exp 1)
     B. Forgetting Factor Analysis (Exp 2)
     C. Perturbation Response (Exp 3)
     D. Multi-Sensor Comparison (Exp 4)
     E. Computational Cost Analysis

VII. Conclusion and Future Work
     - Summary of findings
     - Limitations
     - Future: extend to curved paths, outdoor, sensor fusion with camera

References (20-30 refs)
```

---

## 10. Contributions Tuyên Bố

Bài báo này đóng góp:

1. **Framework self-calibrating online** sử dụng RLS với forgetting factor
   cho xe tự hành quy mô nhỏ, thay thế phương pháp offline LR hiện tại

2. **Phân tích forgetting factor λ** định lượng trade-off giữa
   tốc độ thích ứng và ổn định tham số trên phần cứng thật

3. **Đánh giá so sánh toàn diện** giữa 4 phương pháp (LR, PID, RLS, MPC)
   về yaw error, path deviation, và robustness trước perturbation,
   được validated trên nền tảng Hanback AutoCar III

---

## 12. Rủi Ro & Phương Án Dự Phòng

| Rủi ro | Xác suất | Tác động | Phương án |
|--------|----------|----------|-----------|
| IMU drift quá lớn ở Euler yaw | Trung bình | Dữ liệu không đáng tin | Dùng Gyro-Z (rate) thay vì Euler (absolute); hoặc reset Euler định kỳ |
| MPC quá nặng cho embedded | Trung bình | Không real-time | Giảm horizon N=3, hoặc dùng explicit MPC offline → lookup table |
| AutoCar III hỏng phần cứng | Thấp | Dừng toàn bộ | Backup: mô phỏng trong Python (simulate xe + noise) |
| Không đủ thời gian | Trung bình | Thiếu thí nghiệm | Cắt Exp 4 (multi-sensor) → nice-to-have, không bắt buộc |
| Kết quả RLS không tốt hơn PID | Thấp | Contribution yếu | Focus vào perturbation test (RLS sẽ vượt trội ở adaptability) |

---

## 13. Dependencies & Chuẩn Bị

### Phần mềm cần trên AutoCar III
- [x] Python 3.6+ (đã có)
- [x] `pop` library — Hanback SDK (đã có)
- [ ] `numpy` (kiểm tra version)
- [ ] `cvxpy` hoặc `scipy` — cho MPC (cần install)

### Phần mềm trên PC phân tích
- [ ] Python 3.9+
- [ ] `numpy`, `pandas`, `matplotlib`, `seaborn`
- [ ] `scipy.stats` — statistical tests
- [ ] `LaTeX` distribution — viết paper

### Phần cứng
- [x] Hanback AutoCar III
- [ ] Track thẳng 3m (dán tape trên sàn)
- [ ] Vật nặng 200g (cho perturbation test)
- [ ] Thước đo (path deviation cuối track)
- [ ] Camera ngoài (optional, để quay video demo)

---

## 14. References Chính (Preliminary)

1. Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
2. Haykin, S. (2002). *Adaptive Filter Theory*. Prentice Hall. — Ch. 13: RLS
3. Rajamani, R. (2012). *Vehicle Dynamics and Control*. Springer. — Ch. 2: Bicycle model
4. Kim et al. (2024). "Parameter-Free Adaptive Steering Control with RLS." *Trans. KSAE*.
5. Rawlings, J.B. et al. (2017). *Model Predictive Control: Theory, Computation, Design*. — Lightweight MPC
6. Hanback Electronics. *AutoCar III User Manual & SDK Documentation*.

---

