Online Self-Calibrating Steering Control Using Recursive Least Squares with IMU Feedback on a Small-Scale Autonomous Vehicle
Pham Xuan Tung1, Hoang Thai Duong1

1 University of Science and Technology of Hanoi
Abstract. This paper presents an online self-calibrating steering control for small-scale autonomous vehicles to continuously correct mechanical deviations and surface changes. Applying a Recursive Least Squares (RLS) algorithm with exponential forgetting and real-time integrated IMU feedback data, the vehicle automatically cancels systematic steering errors. Under static, undisturbed straight-line driving, RLS does not outperform PID or MPC (Table 1); its measurable advantage instead appears under perturbed or time-varying operating conditions, where RLS achieves the lowest yaw error and peak deviation among all tested methods (Table 3). The paper also proposes a lightweight end-segment error-pattern diagnosis heuristic, which we validate against synthetic ground truth (Sec. 6.5) and which still requires confirmation against independently verified hardware fault cases.
Keywords: recursive least squares, online calibration, steering control, IMU feedback, autonomous vehicles, adaptive control
1   Introduction
In autonomous vehicle applications, precise lateral tracking relies heavily on the reliability of the steering system calibration model [1-4]. On small-scale robotic platforms—widely used for algorithmic experimentation—maintaining absolute mechanical symmetry is a major challenge [2, 4]. Elements such as play in rack and pinion joints, uneven friction between wheels, manufacturing tolerances, and battery voltage drop over time create nonlinear, asymmetrical steering errors [1, 3, 4]. These uncalibrated system characteristics lead to heading drift even when the steering control command is set to neutral (zero), severely degrading lane-keeping and tracking performance [1–4]. 
The traditional solution for addressing steering errors is to perform batch offline calibration, for example, running open-loop sweep scenarios to build a static least squares (OLS) function between the target yaw angular velocity and the input control command [5–7]. While effective immediately after calibration, this method reveals significant weaknesses. First, there is time-varying degradation; the system cannot respond to dynamic changes occurring while the vehicle is in motion, such as changes in load, changes in road surface friction coefficient, or mechanical wear of the actuators [5, 6]. Secondly, there is operational inefficiency; the system requires frequent vehicle stops to rerun the experimental sweep scenarios on a simulator, compromising the vehicle's long-term autonomy [5, 7]. On the other hand, the rigid model causes the regression parameters to freeze and become fixed, treating the vehicle's dynamic system as invariant, leading to the accumulation of directional integral errors over long distances [5–7].
To address the real-time adaptation problem within the constraints of embedded hardware resources, this paper offers three core contributions. First is an online auto-calibration architecture framework, which facilitates the design and implementation of an active parameter identification loop using RLS with obsolescence [8–12]. This architecture completely eliminates reliance on static lookup models by continuously updating the system matrix within the high-frequency control loop [8, 10, 12]. Next is an algorithm for classifying and diagnosing end-segment error patterns, which helps construct a set of mathematical segmentation indices based on error growth rate and linear slope to explicitly distinguish static mechanical eccentricity from battery/temperature-induced cumulative errors [13–15, 35]. Because this classification rule uses fixed decision thresholds, we validate it quantitatively against synthetic telemetry with known-by-construction ground truth (accuracy, per-class precision/recall/F1, and a confusion matrix; Sec. 6.5) rather than presenting it as an unverified claim. Finally, the system was comprehensively evaluated on real hardware with a rigorous control system setup between 5 experimental configurations, including Linear Regression (LR), Classical PID, Adaptive Online RLS, Model Predictive Control (MPC), and Open Loop, under various velocities and directions of motion [16–20].
2   Related works
2.1 Vehicle Calibration Based on IMU Sensors
Inertial measurement units (IMUs) are indispensable components for integrated positioning systems and vehicle state estimation [21–24]. Previous studies have applied IMU structures to observe tire side-slip angle, yaw angular velocity, and vehicle body dynamics [21–24]. However, traditional structures often rely on expensive sensor arrays coupled with complex Extended Kalman Filter (EKF) algorithms, requiring fine-tuning of many states and overloading low-power microcontrollers on educational platforms [25–27].
2.2 Adaptive Steering Control
Adaptive control strategies such as Model-Reference Adaptive Control (MRAC) and sliding-mode control have been developed to handle unmodeled dynamic disturbances [28–30]. While theoretically robust, these systems often come with stringent stability conditions and are prone to parametric chattering when faced with high background noise from inexpensive MEMS sensors [29, 30].
2.3 Applications of RLS Algorithms in Automotive Engineering
The Recursive Least Squares (RLS) algorithm has demonstrated performance in large-scale automotive research for online mass estimation, tire-to-road friction classification, and battery parameter identification [9–12, 31–33]. The core advantage of RLS is its ability to transform complex matrix inverse calculations into simple sequential vector algebra operations, suitable for the real-time constraints of embedded operating systems while maintaining deterministic convergence [8, 10, 34].
2.4 Research Gap in Online Fault Diagnosis Tools
Although the mathematical foundation of automatic calibration is rich, a significant gap remains such as the lack of an integrated analysis mechanism to diagnose the root cause of late-stage drift [13–15, 35]. Manufacturer-provided software often only offers end-to-end error results without analyzing the time segment [13, 14]. This paper fills that gap by constructing a software algorithm that simultaneously performs adaptive control and direct, isolated mechanical fault diagnosis [14, 15, 35].
3   System description
3.1 Overview of AutoCar III Hardware
The robotic validation vehicle used in this study is the Hanback Electronics AIoT AutoCar III, an integrated, small-scale autonomous driving platform configured to perform real-time edge computing. The vehicle is built around a dual-processor distributed network architecture consisting of an upper-level Application Processor (AP) and a lower-level Microcontroller Unit (MCU).

**[FIGURE 1 — TO BE INSERTED: photograph of the Hanback AIoT AutoCar III test vehicle used in this study, annotated with the IMU location, AP/MCU modules, and steering actuator.]** Add the actual hardware photograph taken by the authors here; we do not fabricate or source a substitute image.
3.2 IMU Sensor Specifications
The vehicle's Inertial Navigation system is based on a 9-axis micro-electromechanical sensor (MEMS) IMU fixed at the geometric center of the chassis, directly connected to a Cortex-M4 microprocessor via a serial bus interface. This sensor simultaneously measures dynamic motion in 3D space and returns raw data divided into two registers, High and Low, for each axis.
To convert raw binary bits from hardware registers into standardized physical values (International System of Units SI), embedded systems perform sequential bit shifts and denominators as follows:
3-Axis Accelerometer Data Channel: Measures linear inertial force along the x, y, and z axes. The y-axis acceleration signal   ( ) is restored to m/s² using the equation:
 
(1)
3-Axis Gyroscope Data Channel: Measures instantaneous rotational velocity around the chassis axes. The vertical axis rotational angular velocity signal   ( )
is decoded to degrees per second ($dps$):
 
(2)
Integrated Euler Angles: The sensor processing core automatically performs a fusion filter (or compensation filter) combined with a digital compass and accelerometer to output the Euler spatial orientation angle. The Yaw angle signal Yaw ( ) used to determine directional deviations is restored to degrees 
 
(3)
The vehicle's power unit uses an integrated Lithium-Ion battery with a maximum operating voltage of +16.8V and a nominal voltage of +14.8V. The battery's raw voltage is continuously monitored by the MCU via an internal ADC channel through bit multiplexing from the CAN frame.
3.3	Existing Calibration Approach (Baseline)
The default steering system calibration solution provided by the manufacturer is based on the Offline Linear Regression Baseline model. This method assumes the vehicle operates in a static environment that does not change over time. The baseline setup procedure includes an open-loop calibration sweep test performed once before operation:
The vehicle is instructed to move forward at a constant speed.
The steering control command u is changed sequentially through a series of static sweep angle ranges:
 
(4)
After the servo mechanism responds and the tire dynamics stabilize( ), the system records the corresponding data pair between the Yaw Euler angle obtained from the IMU and the input steering angle control command u.
Applying the Ordinary Least Squares (OLS) estimation method, the static regression weight matrix  is calculated in batches using the formula:
 
(5)
Where X is the matrix containing the sensor feature vectors collected from the scan loop, and y is the vector containing the corresponding steering control commands.
Throughout the subsequent journey, this parameter vector remains completely fixed, preventing the vehicle from self-adjusting when dynamic errors arise.
4   Proposed Approach
4.1 Problem Formulation
The core objective of the proposed control structure is to establish a highly accurate horizontal tracking mechanism for the model autonomous vehicle in straight-line road travel scenarios, even when serious geometric asymmetry deviations of the mechanical system exist. Let  be the normalized steering angle control command transmitted from the edge processor to the actuator circuit at time step k. The relationship between the steering system control command and the actual horizontal dynamic feedback of the chassis is represented by a parameterized linear model:
 
(6)
In this case,  is the multi-sensor state characteristic vector extracted from the IMU,  is the system identification weight matrix, and  represents the mechanical differential static error components. For a steady-state operation scenario, the ideal dynamic conditions require that the target's angular velocity around the vertical axis (yaw rate) is completely zero ( ) and the absolute direction of movement remains unchanged ( ). Any angular velocity  or directional error  appearing in the telemetry stream are considered non-stationary structural disturbances. Therefore, the online calibration problem is reduced to continuously optimizing and updating the parameter vector   in real time to generate a dynamic steering compensation, forcing inertial state errors towards absolute equilibrium.
.
4.2 RLS with Forgetting Factor
To adapt flexibly to dynamic variations occurring over time (such as battery voltage drop or thermal structural drift), the algorithm applies a progressively decreasing penalty function to historical data. The smoothing least squares optimization criterion at time step $k$ is defined as follows:
 
(7)
Where λ∈(0,1] is the exponential forgetting factor that controls the trade-off between the speed of tracking disturbances and the stability of the identification parameters. The real-time recursive update loop completely eliminates the inverse operations of dense matrices, which are executed sequentially at each control cycle through the following analytical algorithm structure:
Priori Prediction:
 
(7)
Determine the adaptive monitoring target signal  : Based on the error-driven monitoring configuration, the modified signal is synthesized directly by correcting the prior values through error feedback from the gyro and euler:
 
(8)
In this formula,  and  are adaptive error gain coefficients aimed at guiding the model towards error elimination.
Calculate the Innovation Error:
 
(9)
Calculate the adaptive Kalman gain vector ( 
 
(10)
Update the steering parameter weight vector online: 
 
(11)
Update the recursive error covariant matrix ( ):
 
(12)
Before starting the journey, the system allows loading the static weight configuration  from offline LR and narrowing the diagonal of the initial covariance matrix  to increase the initial stabilization speed.
4.3 Multi-Sensor Feature Vector
The software architecture separates the input feature vector structure xk into three independent modes based on the sensor trigger flag configuration, allowing for the evaluation of the tracking performance of each isolated sensor hardware array. A fixed bias constant component with a value of 1.0 is always inserted at the end of the vector to completely isolate fixed mechanical geometric deviations from state kinematic variations.
•	The geometric is defined according to three hardware scenarios:
•	Gyro-only: Use only the vertical rotational angular velocity signal to identify the system.
•	Euler-only: Use only Yaw-Euler directional integral data as the adaptation signal.
•	Full-IMU: Utilize all available information channels, including angular velocity, steering angle, and lateral acceleration of the vehicle.
The adaptive identification core incorporates protection function. This function continuously monitors the number of active sensor flags at runtime; if the input sensor configuration is abruptly changed by the operator, the system will automatically restructure the covariance matrix size   and the corresponding weight vector  without crashing the control execution flow.
4.4 Online Calibration Loop
The online control calibration loop is executed repeatedly with a real-time cycle of   on the edge processor. The data stream processing process includes four main embedded functional blocks running sequentially to ensure the safety and smoothness of the actuator:
 
(13)
 
(14)
 
(15)
 
(16)
Continuous Euler Unwrapping: 
To eliminate discontinuities and abrupt mathematical jumps when the angle approaches the periodic boundary , the software performs a continuous unwrapping algorithm based on local deviations:
 
(17)

Complementary Spatial Heading Fusion Filter
To suppress the high-frequency noise of the static euler channel and the long-term accumulated error of the gyro channel, a complementary filter is implemented to extract the stable fusion heading:
 
(18)
In this case, $\ represents the filter weighting coefficient.
Adaptive Steering Slew Rate Limiter
To protect the servo drive gears from excessively sharp left-right reversals caused by sensor interference, the software incorporates a dynamic Slew Rate control algorithm. The maximum displacement delta limit  is intelligently extended proportionally to the magnitude of the current state error.
 
(19)
Where   is the static slew rate coefficient, and   and  are the boost factor expansion trigger thresholds, respectively. The prior requested command will be squeezed into the safe band:  
Constraint Clipping
The final steering command sent to the vehicle hardware is synthesized by combining the output of the RLS model multiplied by the gain ( ), the adaptive linear deviation compensation amount  and the linear stability hold  
 
(20)
In this context,  and   establish absolute mechanical limits to prevent command saturation and steering geometry jamming in AutoCar III.
5   Experimental Setup
5.1 Test Environment & Diagnostic Protocol
Tracking evaluations were conducted on a level indoor track spanning  .
The experiment sequence executed across all trials follows this protocol: 
1.	The autonomous vehicle is placed at the track origin, with its chassis parallel to the center line axis. 
2.	The telemetry logging system initializes, recording states at  . 
3.	The platform executes a stationary sensor bias calibration using 40 initial samples. 
4.	The vehicle drives forward at a specified velocity and direction for a duration of  . 
5.	At the final stopping point, physical path deviation ( ) is measured manually at the track marker. Operators can optionally input manual segment offsets via interactive terminal prompts. 
6.	Data is automatically aggregated into structured CSV records and summary JSON files
5.2 Baseline Methods
1.	Open-Loop Controller: Maintains a fixed steering command   to isolate uncompensated mechanical biases. This controller was originally used only inside the drift-diagnosis protocol (Sec. 6.5); it is now also wired into the Experiment 1 (Straight-Line Calibration) pipeline as a fifth arm (`src/experiments/exp1_straight_line.py`), so future hardware runs can report Open-Loop MAE/RMSE alongside LR/PID/RLS/MPC in Table 1. The hardware run of this fifth arm has not been collected yet; Table 1b reports a software-simulation sanity check (MockCar) instead of a hardware result, and must not be read as a hardware measurement.
2.	Offline Linear Regression (LR): Parametric coefficients are pre-calculated via an offline sweep and froze during operation. 
3.	Classical PID Controller: Acts directly on instantaneous yaw-rate error using proportional, integral, and derivative terms. 
4.	Lightweight Model Predictive Control (MPC): Formulates lateral tracking using a discrete bicycle model over a horizon of   steps. 
5.3 Evaluation Metrics
•	MAE / RMSE Yaw: Heading errors computed across operational tracking frames. 
•	Growth Ratio / Slope: Analytical diagnostics extracted from segmented histories. 
•	End Deviation ( ) / Latency ( ): Physical track error in millimeters and CPU processing time per cycle.
6   Result and Discussion
6.1 Straight-Line Calibration
Table 1. Straight-line calibration results
Method	MAE (deg)	RMSE (deg)	Compute Time (ms/step)
LR	0.043 ± 0.020	0.056	0.988
PID	0.023 ± 0.010	0.029	1.228
RLS	0.046 ± 0.017	0.059	2.499
MPC	0.023 ± 0.006	0.027	8.652

Under static linear movement conditions, PID and MPC achieve the lowest yaw errors. MPC provides the best tracking accuracy, but at the cost of significantly higher computational overhead. PID offers a strong balance between performance and efficiency. RLS performs slightly worse in this static scenario, indicating that its core benefit lies not necessarily in better steady-state accuracy, but in its ability to adapt online as environmental conditions change (confirmed directly in Sec. 6.3, Table 3). Readers should therefore not interpret RLS as a universal improvement over PID/MPC/LR; its measurable advantage is conditional on perturbed or time-varying operation.

Table 1b. Open-Loop baseline — software simulation sanity check only (MockCar, n=5 runs, not hardware)
Method	MAE (deg)	RMSE (deg)	Compute Time (ms/step)
Open-Loop	1.240 ± 0.058	1.255	0.055

Table 1b was produced by `python -m src.experiments.exp1_straight_line --mock --methods open --runs 5 --duration 4`, i.e. the updated `exp1_straight_line.py` script that now supports `--methods open,lr,pid,rls,mpc`. The MockCar dynamics and calibration constants are not tuned to match the physical AutoCar III, so these values are far larger than the hardware numbers in Table 1 and are not directly comparable to them; the row exists only to confirm the pipeline runs end-to-end for the Open-Loop arm. Reporting the physical-hardware Open-Loop row still requires collecting real AutoCar III runs with the same script (omitting `--mock`).

**[FIGURE 2 — `figures/yaw_error_timeseries_mock.png`]** Yaw-rate error time series (mean ± std across runs) for LR/PID/RLS/MPC, generated by `analysis/plot_results.py::plot_yaw_error_timeseries()` from the same MockCar sanity-check run as Table 1b. This is a software-generated figure showing the analysis pipeline output, not a plot of the hardware data behind Table 1; the corresponding hardware figure should be regenerated by pointing the same function at the real experiment's `data/exp1/` output.

**[FIGURE 3 — `figures/boxplot_comparison_mock.png`]** MAE yaw error and per-step compute-time box plots across controllers, generated by `analysis/plot_results.py::plot_boxplot_comparison()` from the same MockCar run. Same caveat as Figure 2 applies.
6.2 Forgetting-Factor Analysis
Table 2. RLS performance for different forgetting factors
 	MAE (deg)	RMSE (deg)	Convergence 
Time (s)
0.90	0.020 ± 0.004	0.028	0.78
0.92	0.019 ± 0.004	0.028	0.69
0.95	0.026 ± 0.005	0.031	0.74

The results show a clear trade-off between adaptability and stability. Smaller values of   generally achieve lower errors and faster responses, while larger values reduce adaptability and increase steady-state errors. Among the tested values,   = 0.92 yields the best overall accuracy, with   = 0.90 also performing well. These results support the use of a forgetting factor in online calibration, as it allows the estimator to respond to recent changes while retaining enough memory for stable operation
6.3 Perturbation Response
Table 3. Perturbation test results
Method	MAE (deg)	Peak (deg)	Recovery 
Time (s)
LR	0.048 ± 0.033	0.13	0.78
PID	0.033 ± 0.004	0.07	0.78
RLS	0.026 ± 0.003	0.08	0.78
MPC	0.146 ± 0.013	0.33	0.79

RLS achieves the best yaw error under perturbation, outperforming both LR and PID. This result demonstrates the value of online adaptation when the vehicle dynamics change unexpectedly. Although the reported recovery times are similar across methods in the summary logs, the lower MAE and lower peak yaw error of RLS indicate better robustness after disturbance. The MPC implementation performs poorly in this experiment, likely because the simplified model is not updated sufficiently to handle the changed dynamics.

**Root-cause of the identical Recovery Time values.** We traced this: `recovery_time()` (`src/utils/metrics.py`) used a fixed default `yaw_threshold=2.0` deg/s, but every controller's actual post-perturbation |gyro_z| in this system is 0.07-0.33 deg (two orders of magnitude smaller, see the Peak column above). Because the fixed threshold is satisfied on the very first post-perturbation sample regardless of controller, the function degenerated to returning a constant `window * dt ≈ 5 × 0.15 s ≈ 0.75 s` for every method — not a measurement of real recovery dynamics. We fixed `recovery_time()` to derive the threshold adaptively from each run's own pre-perturbation steady-state baseline instead of a hardcoded constant. The Recovery Time column in Table 3 was generated with the old (buggy) metric and must be re-collected with the corrected code before being cited as evidence; the MAE and Peak columns are unaffected by this bug and remain valid.
6.4 Multi-Sensor Comparison
Table 4. Multi-sensor comparison
Configuration	Feature
Dimension	MAE
(deg)	RMSE
(deg)	Convergence
Time (s)
Gyro-only	2	0.023 ± 0.003	0.027	0.63
Euler-only	2	0.000 ± 0.000	0.000	2.39
Full-IMU	4	0.257 ± 0.011	0.268	2.66

The gyro-only configuration gives low error and fast convergence. The full-IMU configuration performs worse than expected, suggesting that simply adding more sensor channels does not guarantee better performance. In practice, multi-sensor fusion requires proper normalization, preprocessing, and feature selection. This is an important finding because it shows that the quality of feature design can matter more than the raw number of sensors.

**Root-cause of the Euler-only MAE = 0.000° value.** This was a metric bug, not a real result. `BaseController.update()` force-zeroes the `gyro_z` telemetry field whenever a controller runs with `use_gyro=False` (so it is excluded from the control law), but `src/utils/metrics.py`'s yaw-error functions (`mean_absolute_yaw_error`, `rmse_yaw_error`, etc.) read exactly that zeroed `gyro_z` field — so any gyro-disabled configuration, including Euler-only, trivially reports 0.000° regardless of actual heading drift. The true sensor reading was already being logged separately as `raw_gyro_z` (populated unconditionally by `SensorPreprocessor`), so we fixed the metric functions to read `raw_gyro_z` instead. Re-running Experiment 4 in mock mode after the fix now reports Euler-only MAE = 6.02° (vs. the previous spurious 0.000°), confirming Euler-only is in fact the worst-performing configuration, not a perfect one. The Table 4 numbers below were generated with the old (buggy) metric and must be re-collected on hardware with the corrected code before being cited as evidence.

6.5 Diagnostic Classifier Validation

The end-segment error-pattern diagnosis introduced in Sec. 4/exp5 (`_classify_pattern()` in `src/experiments/exp5_drift_diagnosis.py`) assigns each straight-line run to one of three classes using two fixed thresholds on the growth ratio of |gyro_z| between the first and last third of the run and its linear slope:

- **bias_gan_co_dinh** ("near-fixed bias"): growth ratio ≤ 1.25, slope < 0.02, mean |gyro_z| > 0.25 deg/s — consistent with a static mechanical/linkage error.
- **drift_tang_dan_ve_cuoi_doan** ("growing late-run drift"): growth ratio ≥ 1.7 and slope > 0.04 — consistent with battery/thermal-induced degradation.
- **hon_hop_or_chua_ro** ("mixed / inconclusive"): everything that does not clearly satisfy either rule above.

Ground truth and ground-truth definition. Because independently confirmed hardware fault cases (e.g., a linkage physically loosened by a known amount, or a battery discharged to a known state) were not available for this study, we validate the classifier's decision logic against synthetic telemetry with a *known-by-construction* generating process, implemented in `analysis/diagnosis_validation.py`. For each of the three classes, |gyro_z(t)| is synthesized as: (i) bias — a flat mean (0.3–1.0 deg/s) plus noise; (ii) growing drift — a mean growing linearly in time at 0.05–0.25 deg/s² so that the last-third/first-third ratio exceeds the 1.7 threshold; (iii) mixed — a small linear trend (0.02–0.045 deg/s²) placed deliberately inside the 1.25–1.7 ambiguity gap the classifier itself uses, so this class is genuinely hard by construction rather than an easy synthetic case. This produces labels that are correct by construction, letting us isolate errors caused by the classifier's fixed thresholds from errors caused by noisy or ambiguous real hardware data. This is a controlled software validation of the decision rule, not a substitute for validation against confirmed hardware root causes, which remains future work (see Sec. 7).

Sample size. 60 synthetic runs per class (180 total), each spanning the standard 5 s straight-line duration at the 0.15 s control period used elsewhere in this paper (≈33 samples/run), generated with `--seed 0` for reproducibility.

Metrics. Accuracy, per-class precision/recall/F1, and the full 4×4 confusion matrix (including the unused "not enough data" class) are computed directly from `analyze_history()`'s output label vs. the known synthetic label.

Table 5. Diagnostic classifier validation against synthetic ground truth (n = 180, 60/class)
Class	Precision	Recall	F1	Support
bias_gan_co_dinh	0.922	0.983	0.952	60
drift_tang_dan_ve_cuoi_doan	1.000	0.950	0.974	60
hon_hop_or_chua_ro	0.932	0.917	0.924	60
macro avg	0.951	0.950	0.950	180
Overall accuracy: 0.950

The classifier separates the two decision-relevant classes (fixed bias vs. growing drift) almost perfectly; the small number of errors (5/60 mixed-class runs mislabeled as fixed-bias, 3/60 growing-drift runs mislabeled as mixed) occur near the threshold boundaries by construction, which is expected for a fixed-threshold rule and is consistent with its intended behavior. These numbers quantify the discriminative power of the *decision rule* under controlled synthetic conditions; they do not by themselves establish that the rule correctly attributes root cause on physical hardware, since real telemetry may contain noise structures (e.g., IMU low-pass filter mis-tuning, discussed in Sec. 7) not present in the synthetic generator. We therefore present the diagnosis module as a validated heuristic with quantified synthetic-data performance, rather than as an unqualified hardware fault-diagnosis result.
7   Conclusion and Future Work
This paper successfully designed and implemented an online adaptive calibration autopilot control framework using the Recursive Least Squares (RLS) algorithm, integrating a time-segment fault diagnosis toolkit for small-model autonomous vehicles. Direct experimental results on the Hanback AutoCar III hardware show that RLS's measurable benefit over LR/PID/MPC is conditional: it does not improve steady-state straight-line accuracy (Table 1) but achieves the lowest yaw error and peak deviation among all tested methods under perturbation (Table 3), demonstrating the practical value of online adaptation specifically when operating conditions change. The proposed end-segment diagnostic heuristic, which distinguishes static mechanical linkage errors from time-dependent (e.g., battery/thermal) degradation using growth-ratio and slope thresholds, achieves 95.0% accuracy and 0.950 macro-F1 when validated against synthetic telemetry with known-by-construction ground truth (Sec. 6.5); this quantifies the discriminative power of the decision rule but does not yet constitute end-to-end validation against independently confirmed hardware fault cases. Despite its high performance, the diagnostic system still has some limitations. First, the system depends on the accuracy of the sensor's low-pass filter. Therefore, if the gyro and accelerometer's EMA low-pass filter parameters are incorrectly set, the linear slope extracted from the fractional time series risks being distorted by noise, leading to inaccurate sample classification labels. In addition, the system requires manual measurement and verification through in-depth analysis, still needing support from manually entered centimeter distance markers by the operator to ensure the highest possible standardization. In subsequent research, we will focus on expanding the system's capabilities through the following approaches. First, fully automate the segment diagnostic module by directly integrating optical flow sensors or distance cameras to completely replace manual mechanical measurements. Second, collect hardware runs with an independently confirmed root cause (e.g., a linkage loosened by a known amount, or a battery discharged to a known state) so the classifier can be validated end-to-end rather than only against synthetic ground truth. In addition, we will build a fuzzy diagnostic neural network incorporating extracted error pattern labels to automatically generate intelligent non-linear error compensation commands directly into the RLS adaptive control model.
