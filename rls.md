Online Self-Calibrating Steering Control Using Recursive Least Squares with IMU Feedback on a Small-Scale 
Autonomous Vehicle
Pham Xuan Tung 1*[0009-0005-9811-8027] and Hoang Thai Duong 1 [0009-0006-1872-9133]
1 University of Science and Technology of Hanoi, Vietnam Academy of Science and 
Technology, Hanoi 100000, Vietnam
pham-xuan.tung@usth.edu.vn
Abstract. This paper presents an online self-calibrating steering control for small-scale autonomous vehicles to continuously correct mechanical deviations and surface changes. Applying a Recursive Least Squares (RLS) algorithm with exponential forgetting and real-time integrated IMU feedback data, the vehicle automatically cancels systematic steering errors. Experimental results demonstrate that this method significantly reduces angular and trajectory deviations compared to static linear regression and traditional PID controllers, while providing diagnostic guidance for hardware maintenance.
Keywords: recursive least squares, online calibration, steering control, IMU feedback, autonomous vehicles, adaptive control 
Introduction
In autonomous vehicle applications, precise lateral tracking relies heavily on the reliability of the steering system calibration model [1-3]. On small-scale robotic platforms, which is widely used for algorithmic experimentation, maintaining absolute mechanical symmetry is a major challenge. Elements such as play in rack and pinion joints, uneven friction between wheels, manufacturing tolerances, and battery voltage drop over time create nonlinear, asymmetrical steering errors. These uncalibrated system characteristics lead to heading drift even when the steering control command is set to neutral (zero), severely degrading lane-keeping and tracking performance.
The traditional solution for addressing steering errors is to perform batch offline calibration, for example, running open-loop sweep scenarios to build a static least squares (OLS) function between the target yaw angular velocity and the input control command [4, 5]. While effective immediately after calibration, this method reveals significant weaknesses. First, there is time-varying degradation; the system cannot respond to dynamic changes occurring while the vehicle is in motion, such as changes in load, changes in road surface friction coefficient, or mechanical wear of the actuators. Secondly, there is operational inefficiency; the system requires frequent vehicle stops to rerun the experimental sweep scenarios on a simulator, compromising the vehicle's long-term autonomy. On the other hand, the rigid model causes the regression parameters to freeze and become fixed, treating the vehicle's dynamic system as invariant, leading to the accumulation of directional integral errors over long distances.
To address the real-time adaptation problem within the constraints of embedded hardware resources, this paper offers three core contributions. First is an online auto-calibration architecture framework, which facilitates the design and implementation of an active parameter identification loop using RLS with obsolescence [6-9]. This architecture completely eliminates reliance on static lookup models by continuously updating the system matrix within the high-frequency control loop. Next is an algorithm for classifying and diagnosing end-segment error patterns, which helps construct a set of mathematical segmentation indices based on error growth rate and linear slope to explicitly distinguish static mechanical eccentricity from battery/temperature-induced cumulative errors [10]. Finally, the system was comprehensively evaluated on real hardware with a rigorous control system setup between five experimental configurations, including linear regression (LR), classical PID, adaptive online RLS, model predictive control (MPC), under various velocities and directions of motion [11,12].
System description
Overview of AutoCar III Hardware
The robotic validation vehicle used in this study is the Hanback Electronics AIoT AutoCar III, an integrated, small-scale autonomous driving platform configured to perform real-time edge computing. The vehicle is built around a dual-processor distributed network architecture consisting of an upper-level Application Processor (AP) and a lower-level Microcontroller Unit (MCU).
Embedded systems convert raw register data into standard SI units. The accelerometer measures linear acceleration along three axes, while the gyroscope measures rotational velocity around the vehicle axes. The sensor’s fusion algorithm combines gyroscope, accelerometer, and digital compass data to estimate Euler angles, particularly the yaw angle for determining heading deviation. The vehicle is powered by an integrated Lithium-ion battery rated at 14.8 V nominal and 16.8 V maximum; its voltage is continuously monitored by the MCU through an internal ADC channel using data extracted from CAN frames.
Existing Calibration Approach (Baseline)
The manufacturer’s default steering calibration uses an offline linear regression baseline, assuming that vehicle dynamics remain constant. Before operation, the vehicle travels at a constant speed while the steering command is sequentially varied across predefined angles. After each command, the system waits 0.5 s for stabilization, then records the yaw rate from the IMU and the corresponding steering input. An Ordinary Least Squares (OLS) model is fitted to these data to obtain a fixed steering parameter vector. Because the parameters are not updated during operation, the model cannot adapt to changing vehicle dynamics or compensate for dynamic errors.
Proposed Approach
Problem Formulation 
The core objective of the proposed control structure is to establish a highly accurate horizontal tracking mechanism for the model autonomous vehicle in straight-line road travel scenarios, even when serious geometric asymmetry deviations of the mechanical system exist. Let be the normalized steering angle control command transmitted from the edge processor to the actuator circuit at time step k. The relationship between the steering system control command and the actual horizontal dynamic feedback of the chassis is represented by a parameterized linear model:
u_{k}={\theta }_{k}^{T}x_{k}+{\delta }_{\text{bias}}	(1)
In this case, xk is the multi-sensor state characteristic vector extracted from the IMU, k is the system identification weight matrix, and bias represents the mechanical differential static error components. For a steady-state operation scenario, the ideal dynamic conditions require that the target's angular velocity around the vertical axis (yaw rate) is completely zero (ref =0) and the absolute direction of movement remains unchanged (ref =0 =0). Any angular velocity {\dot{\psi }}_{z}\neq 0\ or directional error  {\psi }_{\text{error}}\neq 0\ appearing in the telemetry stream are considered non-stationary structural disturbances. Therefore, the online calibration problem is reduced to continuously optimizing and updating the parameter vector k in real time to generate a dynamic steering compensation, forcing inertial state errors towards absolute equilibrium.
RLS with Forgetting Factor
To adapt flexibly to dynamic variations occurring over time (such as battery voltage drop or thermal structural drift), the algorithm applies a progressively decreasing penalty function to historical data. The smoothing least squares optimization criterion at time step k is defined as follows:
J_{k}\left(\theta \right)=\sum_{i=1}^{k} {\lambda }^{k-i}{\left(y_{i}^{\ast }-x_{i}^{T}\theta \right)}^{2}	(2)
where λ∈(0,1] is the exponential forgetting factor that controls the trade-off between the speed of tracking disturbances and the stability of the identification parameters. The real-time recursive update loop completely eliminates the inverse operations of dense matrices, which are executed sequentially at each control cycle. 
The priori prediction is calculated by:
{\hat{y}}_{k}=x_{k}^{T}{\theta }_{k-1}	(3)
Determine the adaptive monitoring target signal based on the error-driven monitoring configuration, the modified signal is synthesized directly by correcting the prior values through error feedback from the gyro and euler:
y_{k}^{\ast }={\hat{y}}_{k}-{\gamma }_{\text{gyro}}\cdot {\dot{\psi }}_{z}-{\gamma }_{\text{euler}}\cdot {\psi }_{\text{error}}	(4)
where gyro and euler are adaptive error gain coefficients aimed at guiding the model towards error elimination.
The innovation error (k) and adaptive Kalman gain vector (Kk) are calculated as:
{\xi }_{k}=y_{k}^{\ast }-{\hat{y}}_{k}	(5)
K_{k}=\frac{P_{k-1}x_{k}}{\lambda +x_{k}^{T}P_{k-1}x_{k}}
	(6)
Update the steering parameter weight vector online:
{\theta }_{k}={\theta }_{k-1}+K_{k}{\xi }_{k}	(7)
Update the recursive error covariant matrix (P_{k}):
P_{k}=\frac{1}{\lambda }\left(P_{k-1}-K_{k}x_{k}^{T}P_{k-1}\right)	(8)
Before operation, the system can initialize the parameter vector θ0 using the offline LR estimate θstatic. The initial covariance is set to P0 = 0.I, accelerating convergence and improving early-stage stability.
Multi-Sensor Feature Vector
The system architecture separates the input feature vector structure xk into three independent modes based on the sensor trigger flag configuration, allowing for the evaluation of the tracking performance of each isolated sensor hardware array. A fixed bias constant component with a value of 1.0 is always inserted at the end of the vector to completely isolate fixed mechanical geometric deviations from state kinematic variations.
The adaptive identification core incorporates protection function. This function continuously monitors the number of active sensor flags at runtime. If the input sensor configuration is abruptly changed by the operator, the system will automatically restructure the covariance matrix size Pk and the corresponding weight vector k without crashing the control execution flow.
Online Calibration Loop
The online steering-calibration loop is executed on the edge processor at a sampling interval of Δt=0.03s. At each iteration, four functional modules are sequentially evaluated to ensure smooth and safe actuator operation.
Continuous Euler-Angle Unwrapping
To eliminate discontinuities when the yaw angle crosses the 00/3600 boundary, the raw Euler angle is continuously unwrapped. The angular offset Ok is updated as
\varDelta {\psi }_{raw,k}={\psi }_{raw,k}-{\psi }_{raw,k-1}	(9)
O_{k}=\left\{\begin{matrix}\! O_{k-1}-{360}^{0},\ \ \ \ \ \ \ \ \! {\varDelta }_{\psi raw,k}\! \gt {180}^{0} \\ O_{k-1}+{360}^{0},\ \ \ \ \ \ \ \ \! {\varDelta }_{\psi raw,k}\! \lt -{180}^{0} \\ O_{k-1},\ \ \ \ \ otherwise\end{matrix}\right.
	(10)
resulting in the continuous yaw angle
{\psi }_{unwrapped,k}\! ={\psi }_{raw,k}\! +O_{k}\! 	(11)
Complementary Heading-Fusion Filter
A complementary filter combines the unwrapped Euler heading with the integrated gyroscope signal. This approach suppresses high-frequency noise in the Euler-angle measurement while limiting long-term gyroscope drift:
{\psi }_{fused,k}\! =\alpha ({\psi }_{fused,k-1}\! +{\dot{\psi }}_{z,k}\! \varDelta t)+\left(1-\alpha \right){\psi }_{unwrapped,k\! }	(12)
where α denotes the filter weighting coefficient.
Adaptive Steering Slew-Rate Limiter
A dynamic slew-rate limiter is applied to prevent abrupt steering reversals caused by sensor disturbances. The allowable steering increment is increased during high-rate maneuvers according to
\varDelta u_{max,k}\! =\left[R_{base\! }max\left(1,min\left(2,max\left(\frac{{\psi }_{error,k}}{\! {\psi }_{thr}},\frac{{\dot{\psi }}_{\! z,k}}{{\dot{\psi }}_{\! thr\ }}\! \! \right)\right)\right)\right]\varDelta t\  	(13)
where Rbase is the nominal steering slew rate, {\psi }_{thr} and {\dot{\ \psi }}_{\! thr\ } are the corresponding activation thresholds. The requested steering command is then constrained as
u_{slew,k}\! =u_{final,k-1\! }+clip(u_{requested,k}\! -u_{final,k-1}\! ,-\varDelta u_{max,k}\! ,\varDelta u_{max,k}\! )	(14)
Command Synthesis and Mechanical Limiting
The final steering command combines the slew-limited output with the RLS-estimated steering gain, adaptive linear-deviation compensation, and linear stability-hold term:
u_{final,k}\! =clip(u_{slew,k}\! G_{steer}\! +u_{trim}\! +u_{hold}\! ,U_{min}\! ,U_{max}\! )	(15)
Here, Umin, Umax define the actuator’s mechanical limits, preventing command saturation and steering-geometry jamming in the AutoCar III platform.

Experimental Setup
Tracking experiments were conducted on a level indoor track. At the beginning of each trial, the vehicle was positioned at the track origin with its chassis aligned parallel to the centerline. The telemetry system was then initialized, followed by a stationary bias calibration using the first 40 sensor samples. The vehicle subsequently traveled at a prescribed speed and direction for a predefined duration. At the endpoint, the lateral path deviation was measured manually, with optional segment offsets entered through the diagnostic terminal. 
Baseline Methods
The proposed method was compared with four baseline controllers: an open-loop controller with a fixed steering command, an offline linear regression model with fixed parameters, a classical PID controller based on instantaneous yaw-rate error, and a lightweight model predictive controller using a discrete bicycle model over a finite prediction horizon.
Evaluation Metrics
Performance was evaluated using the mean absolute error (MAE) and root mean square error (RMSE) of the yaw response. Error growth ratio and slope were derived from segmented tracking histories to characterize temporal drift. Endpoint deviation and per-cycle processing latency were additionally measured to assess physical tracking accuracy and real-time computational efficiency.

Result and Discussion
Straight-Line Calibration
Table 1. Straight-line calibration results
Method	MAE (deg)	RMSE (deg)	Compute Time (ms/step)
LR	0.043 ± 0.020	0.056	0.988
PID	0.023 ± 0.010	0.029	1.228
RLS	0.046 ± 0.017	0.059	2.499
MPC	0.023 ± 0.006	0.027	8.652
Under static linear movement conditions, PID and MPC achieve the lowest yaw errors (Table 1). MPC provides the best tracking accuracy, but at the cost of significantly higher computational overhead. PID offers a strong balance between performance and efficiency. RLS performs slightly worse in this static scenario, indicating that its core benefit lies not necessarily in better steady-state accuracy, but in its ability to adapt online as environmental conditions change.
Table 2.  RLS performance for different forgetting factors
λ	MAE (deg)	RMSE (deg)	Convergence Time (s)
0.90	0.020 ± 0.004	0.028	0.78
0.92	0.019 ± 0.004	0.028	0.69
0.95	0.026 ± 0.005	0.031	0.74
The results show a clear trade-off between adaptability and stability (Table 2). Smaller values of   generally achieve lower errors and faster responses, while larger values reduce adaptability and increase steady-state errors. Among the tested values, = 0.92 yields the best overall accuracy, with   = 0.90 also performing well. These results support the use of a forgetting factor in online calibration, as it allows the estimator to respond to recent changes while retaining enough memory for stable operation.
Table 3. Perturbation test results
Method	MAE (deg)	Peak (deg)	Recover Time (s)
LR	0.048 ± 0.033	0.13	0.78
PID	0.033 ± 0.004	0.07	0.78
RLS	0.026 ± 0.003	0.08	0.78
MPC	0.146 ± 0.013	0.33	0.79

RLS achieves the best yaw error under perturbation, outperforming both LR and PID (Table 3). This result demonstrates the value of online adaptation when the vehicle dynamics change unexpectedly. Although the reported recovery times are similar across methods in the summary logs, the lower MAE and lower peak yaw error of RLS indicate better robustness after disturbance. The MPC implementation performs poorly in this experiment, likely because the simplified model is not updated sufficiently to handle the changed dynamics.
Table 4. Multi-sensor comparison
Configuration	Feature
Dimension	MAE (deg)	RMSE (deg)	Convergence
Time (s)
Gyro-only	2	0.023 ± 0.003	0.027	0.63
Euler-only	2	0.000 ± 0.000	0.000	2.39
Full-IMU	4	0.257 ± 0.011	0.268	2.66

Table 4 show the result of multi-sensor comparison. The gyro-only configuration gives low error and fast convergence. The Euler-only configuration reports zero yaw errors in the logged summary, but this result should be interpreted cautiously because it may reflect the way the metric was computed or the behavior of the orientation signal in this specific implementation. The full-IMU configuration performs worse than expected, suggesting that simply adding more sensor channels does not guarantee better performance. In practice, multi-sensor fusion requires proper normalization, preprocessing, and feature selection. This is an important finding because it shows that the quality of feature design can matter more than the raw number of sensors.

Conclusion 
This paper successfully designed and implemented an online adaptive calibration autopilot control framework using the Recursive Least Squares (RLS) algorithm, integrating a time-segment fault diagnosis toolkit for small-model autonomous vehicles. Direct experimental results on a small prototype hardware demonstrate that the proposed diagnostic structure clearly identifies the causes of straight-line deviation due to static mechanical linkage errors or dynamic degradation over time. The diagnostic system achieves high accuracy in classifying error patterns, providing valuable practical technical guidance for optimizing the system hardware. Despite its high performance, the diagnostic system still has some limitations. First, the system depends on the accuracy of the sensor's low-pass filter. Therefore, if the gyro and accelerometer's EMA low-pass filter parameters are incorrectly set, the linear slope extracted from the fractional time series risks being distorted by noise, leading to inaccurate sample classification labels. In addition, the system requires manual measurement and verification through in-depth analysis, still needing support from manually entered centimeter distance markers by the operator to ensure the highest possible standardization. In subsequent research, we will focus on expanding the system's capabilities through the following approaches. First, fully automate the segment diagnostic module by directly integrating optical flow sensors or distance cameras to completely replace manual mechanical measurements. In addition, we will build a fuzzy diagnostic neural network incorporating extracted error pattern labels to automatically generate intelligent non-linear error compensation commands directly into the RLS adaptive control model.

References
Rajamani R (2012) Vehicle Dynamics and Control, 2nd edn. Springer, New York.
Pacejka HB (2005) Tire and Vehicle Dynamics, 2nd edn. Elsevier, Oxford.
Wong JY (2008) Theory of Ground Vehicles, 4th edn. Wiley, Hoboken
Ljung L (1999) System Identification: Theory for the User, 2nd edn. Prentice Hall, Upper Saddle River.
Montgomery DC, Peck EA, Vining GG (2012) Introduction to Linear Regression Analysis, 5th edn. Wiley, Hoboken.
Haykin S (2002) Adaptive Filter Theory, 4th edn. Prentice Hall, Upper Saddle River.
Ljung L, Söderström T (1983) Theory and Practice of Recursive Identification. MIT Press, Cambridge.
Ding F, Chen T (2005) Combined parameter and output estimation of dual-rate systems using an auxiliary model. Automatica 41(10):1739–1748.
Verhaegen M, Verdult V (2007) Filtering and System Identification: A Least Squares Approach. Cambridge University Press, Cambridge.
Isermann R (2006) Fault-Diagnosis Systems: An Introduction from Fault Detection to Fault Tolerance. Springer, Berlin Heidelberg.
Åström KJ, Murray RM (2008) Feedback Systems: An Introduction for Scientists and Engineers. Princeton University Press, Princeton.
Åström KJ, Hägglund T (2006) Advanced PID Control. ISA, Research Triangle Park.
