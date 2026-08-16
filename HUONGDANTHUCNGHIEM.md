# HƯỚNG DẪN THỰC NGHIỆM AUTOCAR III

## 1. Mục đích

Tài liệu này mô tả:

- cách chuẩn bị phần cứng và phần mềm trước khi chạy thực nghiệm
- quy trình từng bước để chạy các thí nghiệm trên AutoCar III
- cách reset xe, ghi nhận kết quả và lưu dữ liệu
- các lưu ý quan trọng khi chạy trên AutoCar III với Python 3.6.9

## 2. Tổng quan nhanh

Project này có 4 thí nghiệm chính:

1. `exp1_straight_line.py`: chạy thẳng và so sánh LR, PID, RLS, MPC
2. `exp2_lambda_sweep.py`: quét forgetting factor lambda cho RLS
3. `exp3_perturbation.py`: thử nghiệm gây nhiễu tải trọng đột ngột
4. `exp4_multisensor.py`: so sánh các cấu hình cảm biến của RLS

Thông số mặc định trong code:

- tốc độ: `30`
- tần số lặp điều khiển: `10 Hz` (`dt = 0.1 s`)
- số lần lặp mỗi điều kiện: `10`
- thời gian chạy thẳng cho Exp1, Exp2, Exp4: `5 s`
- thời điểm gây nhiễu Exp3: `t = 3 s`
- thời gian sau gây nhiễu Exp3: `5 s`

Nguồn tham chiếu:

- [src/utils/config.py](/D:/AutocarIII_Project/AutocarIII_Project/src/utils/config.py:1)
- [src/experiments/exp1_straight_line.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp1_straight_line.py:1)
- [src/experiments/exp2_lambda_sweep.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp2_lambda_sweep.py:1)
- [src/experiments/exp3_perturbation.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp3_perturbation.py:1)
- [src/experiments/exp4_multisensor.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp4_multisensor.py:1)

## 3. Chuẩn bị phần cứng

Cần chuẩn bị:

- xe Hanback AutoCar III
- pin đủ điện cho toàn bộ buổi đo
- track thẳng dài khoảng `3 m`
- băng dính hoặc vạch tham chiếu để đánh dấu đường tim
- thước đo để đo độ lệch cuối đường
- vật nặng `200 g` cho Exp3
- máy tính để sao chép dữ liệu và phân tích

Yêu cầu mặt sàn:

- sàn phẳng, ít trượt, không có vật cản
- nên giữ cùng một loại mặt sàn cho toàn bộ một nhóm đo để tránh làm thay đổi kết quả

## 4. Chuẩn bị phần mềm trên AutoCar III

### 4.1 Kiểm tra Python

Trên AutoCar III:

```bash
python3 --version
```

Hiện tại trên xe của bạn là `Python 3.6.9`.

### 4.2 Cài thư viện cần thiết

Cần có:

- `pop`
- `numpy`

Kiểm tra nhanh:

```bash
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "from pop import Pilot; print('pop ok')"
```

Nếu chưa có `numpy`:

```bash
pip3 install numpy
```

Tham chiếu:

- [requirements.txt](/D:/AutocarIII_Project/AutocarIII_Project/requirements.txt:1)

### 4.3 Chép source code lên xe

Đặt thư mục project trên xe, ví dụ:

```bash
/home/pi/AutocarIII_Project
```

Đảm bảo trong thư mục này có đầy đủ:

- `src/`
- `tests/`
- `requirements.txt`
- `HUONGDANTHUCNGHIEM.md`

### 4.4 Lưu ý tương thích Python 3.6.9

Hiện tại:

- toàn bộ `exp1`, `exp2`, `exp3`, `exp4` và `run_all_experiments.py` đã được chỉnh để tương thích với Python 3.6.9
- các module hỗ trợ trong `src/controllers`, `src/data_collection`, `src/utils` và `tests/test_controllers.py` cũng đã được vá các điểm không tương thích chính

Vì vậy:

- có thể chạy trực tiếp các experiment trên AutoCar III bằng Python 3.6.9
- vẫn nên test ngắn với `exp1` trước khi chạy các loạt đo dài hơn

## 5. Kiểm tra trước khi chạy

Mỗi lần trước khi bắt đầu một loạt đo:

1. Sạc pin hoặc đảm bảo pin còn đủ.
2. Kiểm tra bánh, servo lái và cảm biến IMU.
3. Đặt xe ở đầu track, hướng thẳng theo vạch tham chiếu.
4. Ghi nhớ rằng xe sẽ tự động `forward()` khi script bắt đầu.
5. Dọn khu vực phía trước xe, không để người đứng ngay trước đầu xe.
6. Mở terminal trong thư mục root của project.
7. Kiểm tra đường dẫn hiện tại:

```bash
pwd
```

8. Nếu cần, tạo thư mục dữ liệu:

```bash
mkdir -p data
```

## 6. Cấu trúc dữ liệu đầu ra

Khi chạy mỗi lần, code sẽ lưu:

- file CSV cho từng run
- file JSON metadata cho từng run
- file JSON tổng hợp cho từng thí nghiệm

Ví dụ với Exp1:

```text
data/
  exp1/
    lr/
    pid/
    rls/
    mpc/
    exp1_summary.json
```

## 7. Quy trình chi tiết cho Experiment 1

### 7.1 Mục tiêu

So sánh 4 controller:

- LR
- PID
- RLS
- MPC

trên bài toán chạy thẳng, đánh giá:

- `MAE yaw`
- `RMSE yaw`
- thời gian tính toán
- độ lệch cuối đường đo thủ công

### 7.2 Lệnh chạy đề nghị

Chạy từ root project:

```bash
python3 -m src.experiments.exp1_straight_line
```

Có thể chạy bằng đường dẫn file:

```bash
python3 src/experiments/exp1_straight_line.py
```

Không dùng:

```bash
python3 exp1_straight_line.py
```

vì file nằm trong `src/experiments/`.

### 7.3 Các bước thực hiện

1. Đặt xe tại vị trí xuất phát, thân xe song song vạch tham chiếu.
2. Mở terminal trong root project.
3. Chạy lệnh:

```bash
python3 -m src.experiments.exp1_straight_line
```

4. Script sẽ in tiêu đề thí nghiệm.
5. Script sẽ thực hiện `calibrate_bias()` khi xe đang đứng yên.
6. Trong giai đoạn này không được chạm vào xe.
7. Sau khi calibrate xong, script sẽ lần lượt chạy:
   - LR
   - PID
   - RLS
   - MPC
8. Với mỗi controller, script sẽ chạy `num_runs` lần.
9. Mỗi run:
   - xe chạy thẳng trong `5 s`
   - script log `timestamp`, `gyro_z`, `euler_yaw`, `accel_y`, `steering`
   - kết thúc run, script dừng lại
10. Sau mỗi run:
   - đo độ lệch cuối đường bằng thước
   - ghi vào sổ tay hoặc bảng Excel
   - đặt lại xe đúng vị trí xuất phát
   - nhấn `Enter` để sang run tiếp theo
11. Lặp lại đến khi hết 4 controller.
12. Sau khi xong, kiểm tra file tổng hợp `data/exp1/exp1_summary.json`.

### 7.4 Mẫu bảng ghi tay cho Exp1

Nên lập bảng ngoài để ghi thêm giá trị đo thủ công:

| Run | Controller | Path deviation (cm) | Ghi chú |
|---|---|---:|---|
| 1 | LR |  |  |
| 2 | LR |  |  |
| 1 | PID |  |  |
| 1 | RLS |  |  |
| 1 | MPC |  |  |

### 7.5 Cách chạy nhanh để test trước khi đo thật

Nếu muốn kiểm tra script có vào được vòng chạy hay không, giảm số run:

```bash
python3 -m src.experiments.exp1_straight_line --runs 1 --duration 2
```

Sau khi test xong, quay về thông số thật:

- `--runs 10`
- `--duration 5`

## 8. Quy trình dự kiến cho Experiment 2

### 8.1 Mục tiêu

So sánh ảnh hưởng của `lambda` trong RLS:

- `0.90`
- `0.92`
- `0.95`
- `0.97`
- `0.99`
- `1.00`

Chỉ số cần theo dõi:

- `MAE yaw`
- `RMSE yaw`
- `convergence time`
- `theta stability`

### 8.2 Quy trình thao tác

1. Giữ nguyên track, mặt sàn, pin và vị trí xuất phát.
2. Chạy từng giá trị `lambda` trong cùng một buổi đo nếu có thể.
3. Mỗi giá trị `lambda` chạy `10` lần.
4. Sau mỗi run:
   - đặt lại xe
   - ghi chú bất thường nếu có
   - nhấn `Enter` để sang run tiếp theo
5. Sau khi xong mỗi giá trị `lambda`, cho xe nghỉ ngắn để tránh nóng servo.
6. Thu dữ liệu kết quả vào `data/exp2/`.

### 8.3 Lưu ý quan trọng

Script [src/experiments/exp2_lambda_sweep.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp2_lambda_sweep.py:1) đã được port cho Python 3.6.9. Có thể chạy trực tiếp trên AutoCar III.

Lệnh dự kiến sau khi port xong:

```bash
python3 -m src.experiments.exp2_lambda_sweep
```

Hoặc với lambda tùy chỉnh:

```bash
python3 -m src.experiments.exp2_lambda_sweep --lambdas 0.90 0.95 0.99
```

## 9. Quy trình dự kiến cho Experiment 3

### 9.1 Mục tiêu

Kiểm tra tốc độ hồi phục khi hệ thống bị thay đổi đột ngột.

Nhiễu được tạo bằng cách:

- đặt thêm `200 g` lên một bên xe tại `t = 3 s`

Chỉ số cần theo dõi:

- `recovery time`
- `peak yaw error`
- `steady-state error` sau nhiễu

### 9.2 Quy trình thao tác

1. Đặt xe tại vị trí xuất phát.
2. Chuẩn bị sẵn vật nặng `200 g` ở vị trí dễ thao tác nhanh.
3. Bật script.
4. Trong `3 s` đầu, không động vào xe.
5. Khi script thông báo perturbation:
   - đặt nhanh vật nặng vào đúng vị trí đã quy định
   - không đẩy lệch xe bằng tay
6. Chờ xe chạy hết pha sau perturbation.
7. Kết thúc run:
   - nhấc vật nặng ra
   - đưa xe về vị trí đầu
   - nhấn `Enter` để sang run tiếp theo
8. Lặp lại cho từng controller.

### 9.3 Lưu ý quan trọng

Script [src/experiments/exp3_perturbation.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp3_perturbation.py:1) đã được port cho Python 3.6.9.

Lệnh dự kiến sau khi port xong:

```bash
python3 -m src.experiments.exp3_perturbation
```

## 10. Quy trình dự kiến cho Experiment 4

### 10.1 Mục tiêu

So sánh 3 cấu hình cảm biến của RLS:

1. `Gyro-only`
2. `Euler-only`
3. `Full-IMU`

### 10.2 Quy trình thao tác

1. Giữ nguyên track và điều kiện đo.
2. Chạy lần lượt 3 cấu hình cảm biến.
3. Mỗi cấu hình chạy `10` run.
4. Sau mỗi run:
   - đặt lại xe
   - nhấn `Enter`
5. Sau khi xong, so sánh:
   - `MAE yaw`
   - `RMSE yaw`
   - `convergence time`

### 10.3 Lưu ý quan trọng

Script [src/experiments/exp4_multisensor.py](/D:/AutocarIII_Project/AutocarIII_Project/src/experiments/exp4_multisensor.py:1) đã được port cho Python 3.6.9.

Lệnh dự kiến sau khi port xong:

```bash
python3 -m src.experiments.exp4_multisensor
```

## 11. Thứ tự chạy đề nghị trong phòng thí nghiệm

Nếu mục tiêu là làm dữ liệu thật trên xe, thứ tự đề nghị:

1. Test ngắn với Exp1: `1 run`, `2 s`
2. Chạy Exp1 đầy đủ: `10 run` mỗi controller
3. Kiểm tra lại dữ liệu CSV và JSON
4. Chạy Exp2
5. Chạy Exp3
6. Chạy Exp4

Lý do:

- Exp1 là bài test cơ bản nhất
- nếu Exp1 chưa ổn, không nên chạy tiếp các thí nghiệm phức tạp hơn

## 12. Kiểm tra sau mỗi buổi đo

Sau mỗi buổi thực nghiệm:

1. Copy thư mục `data/` từ xe về máy tính.
2. Kiểm tra có đủ số file CSV và JSON hay không.
3. Kiểm tra file tổng hợp:
   - `data/exp1/exp1_summary.json`
   - `data/exp2/exp2_summary.json`
   - `data/exp3/exp3_summary.json`
   - `data/exp4/exp4_summary.json`
4. Đổi tên thư mục theo ngày đo nếu cần, ví dụ:

```text
data_2026-06-10_session1/
```

5. Ghi lại:
   - mức pin
   - mặt sàn
   - có tải trọng hay không
   - sự cố bất thường

## 13. Checklist vận hành nhanh

Trước khi đo:

- [ ] pin đủ
- [ ] track sạch, không vật cản
- [ ] vạch tham chiếu rõ ràng
- [ ] terminal đang ở root project
- [ ] `numpy` import được
- [ ] `pop` import được
- [ ] xe đặt đúng hướng

Trong khi đo:

- [ ] không chạm vào xe khi đang calibrate bias
- [ ] reset đúng vị trí sau mỗi run
- [ ] ghi tay path deviation
- [ ] ghi chú bất thường

Sau khi đo:

- [ ] kiểm tra file CSV
- [ ] kiểm tra file JSON
- [ ] copy dữ liệu về PC
- [ ] backup thêm 1 bản

## 14. Lệnh hữu ích

Kiểm tra Python:

```bash
python3 --version
```

Kiểm tra NumPy:

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

Chạy Exp1:

```bash
python3 -m src.experiments.exp1_straight_line
```

Chạy Exp1 ngắn để test:

```bash
python3 -m src.experiments.exp1_straight_line --runs 1 --duration 2
```

## 15. Ghi chú cuối

Nếu mục tiêu trước mắt là thu đủ dữ liệu thật trên AutoCar III, nên:

- chốt quy trình Exp1 trước
- xác nhận file log lưu đúng
- sau đó mở rộng sang Exp2, Exp3, Exp4
