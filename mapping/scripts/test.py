from pykalman import KalmanFilter
import numpy as np

A = [[1, 1, 0],
     [0, 1, 1],
     [0, 0, 1]]
H = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
x_0 = np.array([0, 0, 0])
initial_cov = 1*np.ones((3,3))
kf = KalmanFilter(
    transition_matrices = A,
    observation_matrices = H,
    initial_state_mean=x_0,
    initial_state_covariance=initial_cov,
)
measurements = np.asarray([
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0]
    ])  # 3 observations
#kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
print(filtered_state_means)
