"""
Module for estimating ground truth poses for images from GPS and IMU data.
"""
import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from scipy import interpolate
from scipy.interpolate import interp1d, UnivariateSpline, Rbf
from pykalman import KalmanFilter


def load_gps_and_imu_data(gps_data_path, imu_data_path):
    """
    Load GPS and IMU data from given CSV files from the Malaga Urban Dataset.

    :param gps_data_path: Path to CSV-file containing GPS data.
    :param imu_data_path: Path to CSV-file containing IMU data.
    :returns: (gps_data, imu_data) tuple of numpy arrays containing the full GPS and IMU data from the given files.
    """
    gps_full_data = np.genfromtxt(gps_data_path, skip_header=1)
    imu_full_data = np.genfromtxt(imu_data_path, skip_header=1)

    return gps_full_data, imu_full_data


class GroundTruthEstimator():
    """
    Estimates ground-truth positions from provided GPS and IMU data.
    """

    def __init__(self, gps_full_data, imu_full_data, print_kf_progress=False):
        """
        Initialize `GroundTruthEstimator`.

        :param gps_full_data: Numpy array containing GPS data from the Malaga Urban Dataset, as returned by `load_gps_and_imu_data()`.
        :param imu_full_data: Numpy array containing IMU data from the Malaga Urban Dataset, as returned by `load_gps_and_imu_data()`.
        :param print_kf_progress: Whether to print progress information when calculating estimations using a Kalman-Filter/-Smoother.
        :returns: None
        """
        self.gps_full_data = gps_full_data
        self.imu_full_data = imu_full_data

        self.gps_t = self.gps_full_data[:,0]
        self.gps_local = self.gps_full_data[:,8:11]

        if self.imu_full_data is not None:
            self.imu_t = self.imu_full_data[:,0]
            self.imu_acc = self.imu_full_data[:,1:4]
            self.imu_ypr = self.imu_full_data[:,10:13]
            self.imu_ypr_vel = self.imu_full_data[:,4:7]

        self.print_kf_progress = print_kf_progress
        self.kf = None
        self.kf_filtered_means = None
        self.kf_filtered_covariance = None
        self.kf_smoothed_means = None
        self.kf_smoothed_covariances = None
        self.kf_timestamps = None
        self.kf_measurements = None

        # Probabilistic model parameters
        # TODO Adjust
        self.initial_cov_pos = 0
        self.initial_cov_vel = 1000
        self.initial_cov_acc = 1000
        self.model_cov_pos = lambda dt: ((dt)**3)/6
        self.model_cov_vel = lambda dt: ((dt)**2)/2
        self.model_cov_acc = lambda dt: (dt)**1
        self.obs_cov_gps = 0.01
        # Variance of the accelerometer of the IMU is stated as 0.008
        self.obs_cov_imu = 0.008
        #self.obs_cov_imu = 0.372938397578784
        #self.obs_cov_imu = 0
        #self.obs_cov_imu = 10


    def get_position(self, t, method='cubic', bspline_smoothness=10):
        """
        Get a position groune-truth estimate for a given timestamp using different available methods.

        The position is estimated by one of several methods:
        - Interpolation of the GPS data only ('linear', 'quadratic', 'cubic', 'bspline', 'rbf')
        - State estimation by fusing GPS and IMU data using a Kalman filter or smoother ('kms', 'kmf')

        'kmf' method is not fully implemented.

        :param t: Timestamp(s) to get a position estimate for. Can be a single timestamp or a list of timestamps.
        :param method: Method used for ground-truth estimation.
        :param bspline_smoothness: Smoothness parameter for `method` 'bspline'. Ignored if other method is used.
        :returns: 3D vector representing the position ground-truth estimate for the given timestamps or a list of 3D vectors depending on if a single timestamp or a list of timestamps was passed for `t`.
        """
        gps_x = self.gps_local[:,0]
        gps_y = self.gps_local[:,1]
        gps_z = self.gps_local[:,2]

        if method == 'linear' or method == 'quadratic' or method == 'cubic':
            f_x = interp1d(self.gps_t, gps_x, kind=method)
            f_y = interp1d(self.gps_t, gps_y, kind=method)
            f_z = interp1d(self.gps_t, gps_z, kind=method)
            int_x = f_x(t)
            int_y = f_y(t)
            int_z = f_z(t)
        elif method == 'bspline':
            f_x = UnivariateSpline(self.gps_t, gps_x, s=bspline_smoothness)
            f_y = UnivariateSpline(self.gps_t, gps_y, s=bspline_smoothness)
            f_z = UnivariateSpline(self.gps_t, gps_z, s=bspline_smoothness)
            int_x = f_x(t)
            int_y = f_y(t)
            int_z = f_z(t)
        elif method == 'rbf':
            f_x = Rbf(self.gps_t, gps_x)
            f_y = Rbf(self.gps_t, gps_y)
            f_z = Rbf(self.gps_t, gps_z)
            int_x = f_x(t)
            int_y = f_y(t)
            int_z = f_y(t)
        elif method == 'kmf':
            if self.kf_filtered_means is None:
                self.__generate_kf_filter_estimate()
            # TODO Do something similar to 'kms'
        elif method == 'kms':
            if self.kf_smoothed_means is None:
                self.__generate_kf_smoother_estimate()

            kf_means = self.kf_smoothed_means
            kf_x = kf_means[:,0]
            kf_y = kf_means[:,1]
            kf_z = kf_means[:,2]

            f_x = interp1d(self.kf_timestamps, kf_x, kind='cubic')
            f_y = interp1d(self.kf_timestamps, kf_y, kind='cubic')
            f_z = interp1d(self.kf_timestamps, kf_z, kind='cubic')

            int_x = f_x(t)
            int_y = f_y(t)
            int_z = f_z(t)

        return np.array([int_x, int_y, int_z]).T


    # Returns a pose estimate for the vehicle center
    def get_pose(self, t, method='cubic', bspline_smoothness=10):
        """
        Get a pose groune-truth estimate for a given timestamp using different available methods.

        Position is estimated by one of several methods:
        - Interpolation of the GPS data only ('linear', 'quadratic', 'cubic', 'bspline', 'rbf')
        - State estimation by fusing GPS and IMU data using a Kalman filter or smoother ('kms', 'kmf')

        Orientation is estimated in 2D by the tangent at each point (disregarding the z-axis).

        :param t: Timestamp to get a pose estimate for.
        :param method: Method used for position ground-truth estimation.
        :param bspline_smoothness: Smoothness parameter for `method` 'bspline'. Ignored if other method is used.
        :returns: (position, orientation) tuple of the pose estimate.
        """
        if method == 'kmf' or method == 'kms':
            if method == 'kmf' and self.kf_filtered_means is None:
                self.__generate_kf_filter_estimate()
            elif method == 'kms' and self.kf_smoothed_means is None:
                self.__generate_kf_smoother_estimate()

            timestamps_known = self.kf_timestamps
        else:
            timestamps_known = self.gps_t

        # TODO Instead of not accepting these values extend the interpolation before and after the known time range
        # The pose for the first timestamp or preceding timestamps cannot be estimated
        assert(timestamps_known[0] <= np.min(t))
        # The pose for the last timestamp or succeeding timestamps cannot be estimated
        assert(np.max(t) <= timestamps_known[-1])

        epsilon_t = np.min(np.diff(timestamps_known)) / 100
        t_next = t + epsilon_t;
        pos = self.get_position(t, method=method, bspline_smoothness=bspline_smoothness)
        pos_next = self.get_position(t_next, method=method, bspline_smoothness=bspline_smoothness)

        pos_diff = pos_next - pos

        # TODO Currently the orientation is calculated by disregarding the z-axis (so there is really just a yaw angle calculated)
        base_x = np.array([1.0, 0.0, 0.0])

        # Discard z coordinate
        v1 = base_x[0:2]
        v2 = pos_diff[0:2]

        # Calculate angle between v1 and v2
        yaw = np.math.atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))

        orientation_offset = tf.euler.euler2quat(0, np.deg2rad(90), np.deg2rad(-90), 'rxyz')
        orientation = tf.quaternions.qmult(tf.euler.euler2quat(yaw, 0, 0, 'rzyx'), orientation_offset)

        return pos.T, orientation


    def __generate_kalman_data(self):
        """
        Generate data for `KalmanFilter` object and initialize the `KalmanFilter` objects.

        :returns: None
        """
        if self.print_kf_progress:
            print('Generating inputs...')
        self.kf_timestamps, initial_state, initial_covariance, transition_matrices, transition_covariances, observation_matrices, observation_covariances, self.kf_measurements = self.__generate_kalman_input()

        g = 9.81
        self.kf = KalmanFilter(
            n_dim_state = 9,
            n_dim_obs = 9,
            initial_state_mean = initial_state,
            initial_state_covariance = initial_covariance,
            transition_matrices = transition_matrices,
            transition_covariance = transition_covariances,
            observation_matrices = observation_matrices,
            observation_covariance = observation_covariances,
            transition_offsets=np.zeros(9),
            observation_offsets=np.array([0, 0, 0, 0, 0, 0, 0, 0, g]),
        )

        if self.print_kf_progress:
            print('Running EM algorithm...')

        # TODO Not sure if we should use that
        #self.kf = self.kf.em(self.kf_measurements, n_iter=5, em_vars=['transition_covariance', 'observation_covariance'])#, 'initial_state_mean', 'initial_state_covariance'])


    def __generate_kf_filter_estimate(self):
        """
        Run Kalman filter on available GPS and IMU data.

        :returns: None
        """
        if self.kf is None:
            self.__generate_kalman_data()

        if self.print_kf_progress:
            print('Running filter...')
        self.kf_filtered_means, self.kf_filtered_covariance = self.kf.filter(self.kf_measurements)


    def __generate_kf_smoother_estimate(self):
        """
        Run Kalman smoother on available GPS and IMU data.

        :returns: None
        """
        if self.kf is None:
            self.__generate_kalman_data()

        if self.print_kf_progress:
            print('Running smoother...')
        self.kf_smoothed_means, self.kf_smoothed_covariances = self.kf.smooth(self.kf_measurements)


    # The state transition depends on the length of a timestep and we have
    # variable-length timesteps, so we have different transition matrices for
    # different steps
    # TODO Include angular velocity (measured by IMU) in model
    def __get_state_transition_matrix(self, dt):
        """
        Get a state transition matrix for the model used by the Kalman filter/smoother given a time step length.

        :param dt: Time step length to get the transition matrix for.
        :returns: State transition matrix with shape (9,9).
        """
        return np.array([
            [1, 0, 0, dt,  0,  0,  (dt**2)/2,  0,  0],
            [0, 1, 0,  0, dt,  0,  0,  (dt**2)/2,  0],
            [0, 0, 1,  0,  0, dt,  0,  0,  (dt**2)/2],
            [0, 0, 0,  1,  0,  0, dt,  0,  0],
            [0, 0, 0,  0,  1,  0,  0, dt,  0],
            [0, 0, 0,  0,  0,  1,  0,  0, dt],
            [0, 0, 0,  0,  0,  0,  1,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  1,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  1],
        ])


    def __gps_measurement(self, gps_pos):
        """
        Get the obervation matrix and measurement vector for the model used by the Kalman filter/smoother given a GPS position measurement.

        :param gps_pos: 3D vector representing a GPS position measurement.
        :returns: (measurement vector, observation matrix) tuple. Measurement vector has shape (9,), observation matrix has shape (9,9).
        """
        measurement = np.block([gps_pos, 0, 0, 0, 0, 0, 0])
        observation_matrix = np.array([
            [1, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 1, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 1,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            ])
        return measurement, observation_matrix


    def __imu_measurement(self, imu_acc):
        """
        Get the obervation matrix and measurement vector for the model used by the Kalman filter/smoother given a IMU acceleration measurement.

        :param imu_acc: 3D vector representing a IMU acceleration measurement.
        :returns: (measurement vector, observation matrix) tuple. Measurement vector has shape (9,), observation matrix has shape (9,9).
        """
        measurement = np.block([0, 0, 0, 0, 0, 0, imu_acc])
        observation_matrix = np.array([
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  1,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  1,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  1],
            ])
        return measurement, observation_matrix


    def __generate_kalman_input(self, print_progress=False):
        """
        Get all data needed for the Kalman filter/smoother. All available GPS and IMU data is used. First measurement is taken as initial state.

        Kalman filter/smoother state vector is x = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z].T

        :param print_progress: Whether to print progress information.
        :returns: (timestamps, initial_state, initial_covariance, transition_matrices, transition_covariances, observation_matrices, observation_covariances, measurements) tuple containing all data needed to initialize the `KalmanFilter` object.
        """
        def add_element(array, element, i):
            if array is None:
                array = np.zeros((timesteps,) + element.shape)

            array[i] = element
            return array

        cov_gps = self.obs_cov_gps
        cov_imu = self.obs_cov_imu
        obs_cov = np.diag([
            cov_gps, cov_gps, cov_gps,
            0, 0, 0,
            cov_imu, cov_imu, cov_imu
        ])

        time_min = np.min(np.append(self.gps_t, self.imu_t))
        time_max = np.max(np.append(self.gps_t, self.imu_t))
        last_t = 0.0

        timesteps = self.gps_t.size + self.imu_t.size

        initial_state = None
        initial_covariance = None
        transition_matrices = None
        observation_matrices = None
        observation_covariances = None
        transition_covariances = None
        measurements = None
        timestamps = None

        i = 0
        while last_t < (time_min + (time_max-time_min)/1):
            pending_gps = np.where(self.gps_t > last_t)[0]
            pending_imu = np.where(self.imu_t > last_t)[0]
            if len(pending_gps) == 0 and len(pending_imu) == 0:
                break
            if len(pending_gps) > 0:
                gps_next_idx = pending_gps[0]
                gps_t_next = self.gps_t[gps_next_idx]
            else:
                gps_t_next = float('inf')

            if len(pending_imu) > 0:
                imu_next_idx = pending_imu[0]
                imu_t_next = self.imu_t[imu_next_idx]
            else:
                imu_t_next = float('inf')

            # Check which measurement comes next
            if gps_t_next < imu_t_next:
                measurement, observation_matrix = self.__gps_measurement(self.gps_local[gps_next_idx])
                t = gps_t_next
            else:
                # Correct IMU acceleration measurements according to current orientation
                yaw = self.imu_ypr[imu_next_idx,0]
                pitch = self.imu_ypr[imu_next_idx,1]
                roll = self.imu_ypr[imu_next_idx,2]
                R_Ii = tf.euler.euler2mat(yaw, pitch, roll, axes='rzyx')
                acc = np.dot(R_Ii, self.imu_acc[imu_next_idx])

                measurement, observation_matrix = self.__imu_measurement(acc)
                t = imu_t_next

            if last_t == 0.0:
                # Take first measurment as initial state
                initial_state = np.dot(observation_matrix, measurement)
                initial_covariance = np.copy(obs_cov)
                initial_covariance[0:3,0:3] = 0 * np.eye(3)
                #initial_covariance = obs_cov
            else:
                dt = t - last_t
                cov_pos = self.model_cov_pos(dt)
                cov_vel = self.model_cov_vel(dt)
                cov_acc = self.model_cov_acc(dt)

                transition_matrix = self.__get_state_transition_matrix(dt)
                model_cov = np.diag([
                    cov_pos, cov_pos, cov_pos,
                    cov_vel, cov_vel, cov_vel,
                    cov_acc, cov_acc, cov_acc
                ])

                # Store input for smoothing later
                timestamps = add_element(timestamps, last_t, i)
                transition_matrices = add_element(transition_matrices, transition_matrix, i)
                observation_matrices = add_element(observation_matrices, observation_matrix, i)
                observation_covariances = add_element(observation_covariances, obs_cov, i)
                transition_covariances = add_element(transition_covariances, model_cov, i)
                measurements = add_element(measurements, measurement, i)
                i += 1

            assert(last_t < t)
            last_t = t
            progress = 100 * (last_t - time_min) / (time_max - time_min)
            if self.print_kf_progress:
                print('{:.2f}%    '.format(progress), end='\r')

        # Add last timestamp
        timestamps = add_element(timestamps, last_t, i)
        assert(timesteps - 1 == i)

        if self.print_kf_progress:
            print('Done     ')

        # Flatten timestamps array
        return (timestamps, initial_state, initial_covariance, transition_matrices, transition_covariances, observation_matrices, observation_covariances, measurements)





def get_lim_equal_scaling_3d(data_x, data_y, data_z):
    min_x = np.min(data_x)
    min_y = np.min(data_y)
    min_z = np.min(data_z)

    max_x = np.max(data_x)
    max_y = np.max(data_y)
    max_z = np.max(data_z)

    diff_x = np.abs(max_x - min_x)
    diff_y = np.abs(max_y - min_y)
    diff_z = np.abs(max_z - min_z)

    max_range = np.max([diff_x, diff_y, diff_z])

    mid_x = min_x + (diff_x) / 2.0
    mid_y = min_y + (diff_y) / 2.0
    mid_z = min_z + (diff_z) / 2.0

    return [(mid_x-max_range/2.0, mid_x+max_range/2.0), (mid_y-max_range/2.0, mid_y+max_range/2.0), (mid_z-max_range/2.0, mid_z+max_range/2.0)]

def plot_imu_data(t, ypr, ypr_vel):
    ypr_from_vel = integrate.cumtrapz(ypr_vel, t, axis=0)

    fig_imu_data = plt.figure()
    ax = fig_imu_data.add_subplot(211)
    ax.set_title('Integrated rotation velocities')
    ax.plot(t[1:], np.rad2deg(ypr_from_vel[:,0]), color='blue')
    ax.scatter(t, np.rad2deg(ypr_vel[:,0]), s=1, color='lightblue')
    ax.plot(t[1:], np.rad2deg(ypr_from_vel[:,1]), color='red')
    ax.scatter(t, np.rad2deg(ypr_vel[:,1]), s=1, color='salmon')
    ax.plot(t[1:], np.rad2deg(ypr_from_vel[:,2]), color='green')
    ax.scatter(t, np.rad2deg(ypr_vel[:,2]), s=1, color='lightgreen')
    ax.grid()
    ax.legend(['$\int$ yaw rate', 'yaw rate', '$\int$ pitch rate', 'pitch rate', '$\int$ roll rate', 'roll rate'])

    ax = fig_imu_data.add_subplot(212)
    ax.set_title('Orientation estimate from IMU')
    ypr_rel = ypr - ypr[0]
    ax.plot(t, np.rad2deg(ypr_rel[:,0]))
    ax.plot(t, np.rad2deg(ypr_rel[:,1]))
    ax.plot(t, np.rad2deg(ypr_rel[:,2]))
    ax.grid()
    ax.legend(['yaw', 'pitch', 'roll'])
    plt.show(block=False)
    # Pause so that the window gets drawn
    plt.pause(0.0001)

def covariance_magnitude(covariances):
    magnitudes = np.zeros((covariances.shape[0],1))

    for i in range(magnitudes.shape[0]):
        cov_mat = covariances[i]
        magnitudes[i] = np.max(np.diag(cov_mat)[0:3])

    return magnitudes

def plot_gps_interpolation(ax, estimator):
    t = estimator.gps_t
    target_point_count = t.shape[0] * 100
    int_lin_t = np.linspace(t[0], t[-1], num=target_point_count)

    # Linear interpolation
    int_lin_x, int_lin_y, _ = estimator.get_position(int_lin_t, method='linear')
    # Quadratic interpolation
    int_quad_x, int_quad_y, _ = estimator.get_position(int_lin_t, method='quadratic')
    # Cubic interpolation
    int_cub_x, int_cub_y, _ = estimator.get_position(int_lin_t, method='cubic')
    # B-Spline interpolation
    int_bspl_x, int_bspl_y, _ = estimator.get_position(int_lin_t, method='bspline')
    # Radial basis function interpolation
    int_rbf_x, int_rbf_y, _ = estimator.get_position(int_lin_t, method='rbf')

    ax.scatter(int_lin_x, int_lin_y, s=1, color='lightgray')
    ax.scatter(int_quad_x, int_quad_y, s=1, color='salmon')
    ax.scatter(int_cub_x, int_cub_y, s=1, color='lightblue')
    ax.scatter(int_bspl_x, int_bspl_y, s=1, color='lightgreen')
    ax.scatter(int_rbf_x, int_rbf_y, s=1, color='plum')

def plot_state_estimation(estimator, plot_interpoltation=False, dim3=False, visualize_covariance=False):
    gps_x = estimator.gps_local[:,0]
    gps_y = estimator.gps_local[:,1]
    gps_z = estimator.gps_local[:,2]

    kf_mean_fil_x = estimator.kf_filtered_means[:,0]
    kf_mean_fil_y = estimator.kf_filtered_means[:,1]
    kf_mean_fil_z = estimator.kf_filtered_means[:,2]

    kf_mean_smo_x = estimator.kf_smoothed_means[:,0]
    kf_mean_smo_y = estimator.kf_smoothed_means[:,1]
    kf_mean_smo_z = estimator.kf_smoothed_means[:,2]

    cov_mag_fil = covariance_magnitude(estimator.kf_filtered_covariance)
    cov_mag_smo = covariance_magnitude(estimator.kf_smoothed_covariances)

    fig = plt.figure()
    if dim3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if plot_interpoltation:
        # TODO Enable for 3d
        plot_gps_interpolation(ax, estimator)

    #ax.set_title('Ground truth estimate')
    ax.grid()

    if visualize_covariance:
        print(cov_mag_smo.shape)
        print(np.argmin(cov_mag_smo))
        print(np.argmax(cov_mag_smo))
        s_norm_fil = 1000# / np.min(cov_mag_fil)
        s_norm_smo = 10000# / np.min(cov_mag_smo)
        s_fil = np.clip((cov_mag_fil * s_norm_fil)**2, 1, 10000)
        s_smo = np.clip((cov_mag_smo * s_norm_smo)**2, 1, 10000)
    else:
        s_fil = 1
        s_smo = 1

    if dim3:
        #scatter = ax.scatter(kf_mean_fil_x, kf_mean_fil_y, kf_mean_fil_z, s=s_fil, color='salmon')
        scatter = ax.scatter(kf_mean_smo_x, kf_mean_smo_y, kf_mean_smo_z, s=s_smo, color='tab:red')
        ax.scatter(gps_x, gps_y, gps_z, s=2, color='tab:blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        #scatter = ax.scatter(kf_mean_fil_x, kf_mean_fil_y, s=s_fil, color='salmon')
        scatter = ax.scatter(kf_mean_smo_x, kf_mean_smo_y, s=s_smo, color='tab:red')
        ax.scatter(gps_x, gps_y, s=2, color='tab:blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    if dim3:
        limits = get_lim_equal_scaling_3d(gps_x, gps_y, gps_z)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
    else:
        plt.axis('equal')

    #ax.legend(['Kalman smoothing (GPS+IMU)', 'GPS'])

    plt.show(block=True)


# TODO At least two problems remain:
# - The GPS local frames are rotationally aligned. That means the rotation of the GPS local frame is not reset for each extract.
#   So far the assumption was that the GPS local frame always resets for each extract (translationally and rotationally).
# - The orientation measurement of the IMU exhibits drift. That means even when it is correctly aligned with the GPS local frame the linear accelerations will be transformed in the wrong direction.
#   Idea: Use the GPS positions to calculate an orientation and use that to calculate the local (between two GPS measurements) IMU orientation bias, i.e. "reset" the IMU orientation at each GPS measuement.

# TODO Possible improvements:
# - Include orientation in state space model. Could be done by linearizing trigonometric functions around the current state
# - Currently it is assumed that IMU and GPS sensor are at the same position on the car. However, this is not really true. To take that into account one could choose the GPS sensor as the "center" point of the car and transform the IMU measurements to that point.

if __name__ == '__main__':

    timestamps = [
        1261230001.530205,
        1261230001.480201,
        1261230001.430199,
        1261230001.380191,
        1261230001.330212,
        1261230001.280222,
        1261230001.230220,
        1261230001.180217,
        1261230001.130214,
        1261230001.080210
    ]

    gps_full_data, imu_full_data = load_gps_and_imu_data('07/gps.csv', '07/imu.csv')
    estimator = GroundTruthEstimator(gps_full_data, imu_full_data, print_kf_progress=True)
    positions = estimator.get_position(timestamps, method='cubic')
    print(positions)
    diffs = np.diff(positions, axis=0)
    print(diffs)
    norms = np.linalg.norm(diffs, axis=1)
    print(norms)
    cumsum = np.cumsum(norms)
    print(cumsum)
    exit()

    plot_imu_data(estimator.imu_t, estimator.imu_ypr, estimator.imu_ypr_vel)

    estimator.__generate_kf_filter_estimate()
    estimator.__generate_kf_smoother_estimate()

    plot_state_estimation(estimator, plot_interpoltation=True, dim3=False, visualize_covariance=False)




















