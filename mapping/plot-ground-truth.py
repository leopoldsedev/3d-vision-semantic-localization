import numpy as np
import matplotlib.pyplot as plt
import transforms3d as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from pykalman import KalmanFilter

np.set_printoptions(threshold=100, linewidth=200)

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

def plot_gps(x, y, z):
    #ax.scatter(x, y, z, s=10)
    ax.scatter(x, y, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    #limits = get_lim_equal_scaling_3d(x, y, z)
    #ax.set_xlim(limits[0])
    #ax.set_ylim(limits[1])
    #ax.set_zlim(limits[2])

def plot_imu(x, y, z):
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    limits = get_lim_equal_scaling_3d(x, y, z)
    #ax.set_xlim(limits[0])
    #ax.set_ylim(limits[1])
    #ax.set_zlim(limits[2])

def plot_kf(x, y, z, s):
    ax.scatter(x, y, z, s=s, color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    limits = get_lim_equal_scaling_3d(x, y, z)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])

# TODO At least two problems remain:
# - The GPS local frames are rotationally aligned. That means the rotation of the GPS local frame is not reset for each extract.
#   So far the assumption was that the GPS local frame always resets for each extract (translationally and rotationally).
# - The orientation measurement of the IMU exhibits drift. That means even when it is correctly aligned with the GPS local frame the linear accelerations will be transformed in the wrong direction.
#   Idea: Use the GPS positions to calculate an orientation and use that to calculate the local (between two GPS measurements) IMU orientation bias, i.e. "reset" the IMU orientation at each GPS measuement.

gps_full_data = np.genfromtxt('05/gps.csv', skip_header=1)
imu_full_data = np.genfromtxt('05/imu.csv', skip_header=1)

gps_timestamps = gps_full_data[:,0]
gps_geocen_x = gps_full_data[:,12]
gps_geocen_y = gps_full_data[:,13]
gps_geocen_z = gps_full_data[:,14]
gps_geocen = np.block([[gps_geocen_x], [gps_geocen_y], [gps_geocen_z]]).T # size n x 3
gps_local_x = gps_full_data[:,8]
gps_local_y = gps_full_data[:,9]
gps_local_z = gps_full_data[:,10]
gps_local = np.block([[gps_local_x], [gps_local_y], [gps_local_z]]).T # size n x 3
gps_vel_x = np.diff(gps_geocen_x)
gps_vel_y = np.diff(gps_geocen_y)
gps_vel_z = np.diff(gps_geocen_z)
gps_acc_x = np.diff(gps_vel_x)
gps_acc_y = np.diff(gps_vel_y)
gps_acc_z = np.diff(gps_vel_z)

g = 9.81
imu_timestamps = imu_full_data[:,0] # first entry: 1261229981.5854
imu_time_diff = np.diff(imu_timestamps)
imu_acc_x = imu_full_data[:,1]
imu_acc_y = imu_full_data[:,2]
# TODO This is probably not right, because the direction of g is in the global frame, but the IMU data is given in the sensor frame
imu_acc_z = imu_full_data[:,3] - g # z acceleration includes gravity.
imu_yaw_vel = imu_full_data[:,4]
imu_pitch_vel = imu_full_data[:,5]
imu_roll_vel = imu_full_data[:,6]
imu_yaw_rate = imu_full_data[:,4]
imu_yaw = imu_full_data[:,10]
imu_pitch = imu_full_data[:,11]
imu_roll = imu_full_data[:,12]
# Make orientation relative to start
#imu_yaw -= imu_yaw[0]
#imu_pitch -= imu_pitch[0]
#imu_roll -= imu_roll[0]
R_I0 = tf.euler.euler2mat(imu_yaw[0], imu_pitch[0], imu_roll[0], axes='rzyx')


# This is working really badly. Here is the reason: https://stackoverflow.com/questions/26476467/calculating-displacement-using-accelerometer-and-gyroscope-mpu6050
imu_vel_x = integrate.cumtrapz(imu_acc_x, imu_timestamps)
imu_vel_y = integrate.cumtrapz(imu_acc_y, imu_timestamps)
imu_vel_z = integrate.cumtrapz(imu_acc_z, imu_timestamps)
imu_pos_x = integrate.cumtrapz(imu_vel_x, imu_timestamps[1:])
imu_pos_y = integrate.cumtrapz(imu_vel_y, imu_timestamps[1:])
imu_pos_z = integrate.cumtrapz(imu_vel_z, imu_timestamps[1:])

print("Mean yaw vel: {}".format(np.mean(imu_yaw_vel)))
print("Total yaw rate change (deg): {}".format(np.rad2deg(imu_yaw[-1] - imu_yaw[0])))
print("IMU acceleration X (min -- max -- mean): {} -- {} -- {}".format(np.min(imu_acc_x), np.max(imu_acc_x), np.mean(imu_acc_x)))
print("IMU acceleration Y (min -- max -- mean): {} -- {} -- {}".format(np.min(imu_acc_y), np.max(imu_acc_y), np.mean(imu_acc_y)))
print("IMU acceleration Z (min -- max -- mean): {} -- {} -- {}".format(np.min(imu_acc_z), np.max(imu_acc_z), np.mean(imu_acc_z)))
yaw_from_vel = integrate.cumtrapz(imu_yaw_vel, imu_timestamps)
pitch_from_vel = integrate.cumtrapz(imu_pitch_vel, imu_timestamps)
roll_from_vel = integrate.cumtrapz(imu_roll_vel, imu_timestamps)
fig2 = plt.figure()
ax = fig2.add_subplot(211)
ax.plot(imu_timestamps[1:], np.rad2deg(yaw_from_vel), color='blue')
ax.scatter(imu_timestamps, np.rad2deg(imu_yaw_vel), s=1, color='lightblue')
ax.plot(imu_timestamps[1:], np.rad2deg(pitch_from_vel), color='red')
ax.scatter(imu_timestamps, np.rad2deg(imu_pitch_vel), s=1, color='salmon')
ax.plot(imu_timestamps[1:], np.rad2deg(roll_from_vel), color='green')
ax.scatter(imu_timestamps, np.rad2deg(imu_roll_vel), s=1, color='lightgreen')
plt.grid()
ax.legend(['$\int$ yaw rate', 'yaw rate', '$\int$ pitch rate', 'pitch rate', '$\int$ roll rate', 'roll rate'])

ax = fig2.add_subplot(212)
ax.plot(imu_timestamps, np.rad2deg(imu_yaw))
ax.plot(imu_timestamps, np.rad2deg(imu_pitch))
ax.plot(imu_timestamps, np.rad2deg(imu_roll))
plt.grid()
ax.legend(['yaw', 'pitch', 'roll'])

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
plt.axis('equal')
plot_gps(gps_local_x, gps_local_y, gps_local_z)
#plot_imu(imu_pos_x, imu_pos_y, imu_pos_z)
plt.grid()
plt.show()


# Kalman filter with constant acceleration model and GPS only
# State vector is x = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z].T
# TODO Include angular velocity (measured by IMU) in model
state_transition = lambda dt: np.array([
    [1, 0, 0, dt,  0,  0,  0,  0,  0],
    [0, 1, 0,  0, dt,  0,  0,  0,  0],
    [0, 0, 1,  0,  0, dt,  0,  0,  0],
    [0, 0, 0,  1,  0,  0, dt,  0,  0],
    [0, 0, 0,  0,  1,  0,  0, dt,  0],
    [0, 0, 0,  0,  0,  1,  0,  0, dt],
    [0, 0, 0,  0,  0,  0,  1,  0,  0],
    [0, 0, 0,  0,  0,  0,  0,  1,  0],
    [0, 0, 0,  0,  0,  0,  0,  0,  1],
    ])
x_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
# TODO Adjust covariances
initial_cov_pos = 0
initial_cov_vel = 10
initial_cov_acc = 10
model_cov_pos = 100
model_cov_vel = 10
model_cov_acc = 1
obs_cov_gps = 10 # TODO Change this
# Variance of the accelerometer of the IMU is stated as 0.008
#obs_cov_imu = 0.372938397578784 # TODO Don't forget to revert this
obs_cov_imu = 0
initial_cov = np.block([
    [initial_cov_pos * np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)), initial_cov_vel * np.eye(3), np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3)), initial_cov_acc * np.eye(3)],
    ])
model_cov = np.block([
    [model_cov_pos * np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)), model_cov_vel * np.eye(3), np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3)), model_cov_acc * np.eye(3)],
    ])
obs_cov = np.block([
    [obs_cov_gps * np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3)), obs_cov_imu * np.eye(3)],
    ])

kf = KalmanFilter(
    #transition_matrices = state_transition(1),
    #observation_matrices = observation_matrix,
    n_dim_state=9,
    n_dim_obs=9,
    initial_state_mean=x_0,
    initial_state_covariance=initial_cov,
    #observation_covariance=cov_obs,
    #transition_covariance=cov_trans
)

timestamp = np.min([gps_timestamps[0], imu_timestamps[0]])
steps = len(gps_timestamps) + len(imu_timestamps)
current_filtered_state_mean = x_0
current_filtered_state_covariance = initial_cov
state_means = np.array([current_filtered_state_mean])
state_covs = np.array([current_filtered_state_covariance])
print("t_0 = {}".format(timestamp))
print("x_0 = {}".format(current_filtered_state_mean))

to_process = len(gps_timestamps) + len(imu_timestamps) - 1000
for i in range(int(to_process)):
    pending_gps = np.where(gps_timestamps > timestamp)[0]
    pending_imu = np.where(imu_timestamps > timestamp)[0]
    if len(pending_gps) > 0:
        gps_next_idx = pending_gps[0]
        gps_next = gps_timestamps[gps_next_idx]
    else:
        gps_next = 0.0

    if len(pending_imu) > 0:
        imu_next_idx = pending_imu[0]
        imu_next = imu_timestamps[imu_next_idx]
    else:
        imu_next = 0.0

    # Check which measurement comes next
    if gps_next < imu_next:
        measurement = np.block([gps_local_x[gps_next_idx], gps_local_y[gps_next_idx], gps_local_z[gps_next_idx], 0, 0, 0, 0, 0, 0]).T
        print("GPS update: {}".format(measurement))
        observation_matrix = np.array([
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0, 0,  0,  0,  0,  0,  0,  0],
            ])

        timestamp_next = gps_next
    else:
        # TODO There either there is something wrong with the IMU data interpretation or the linear acceleration measurement is really bad
        # Correct IMU acceleration measurements according to current orientation
        R_Ii = tf.euler.euler2mat(imu_yaw[imu_next_idx], imu_pitch[imu_next_idx], imu_roll[imu_next_idx], axes='rzyx')
        R_0i = np.dot(R_I0.T, R_Ii)
        #R_0i = tf.euler.euler2mat(0, 0, 0, axes='rzyx')
        #R_0i_alt = tf.euler.euler2mat(imu_yaw[imu_next_idx]-imu_yaw[0], imu_pitch[imu_next_idx]-imu_pitch[0], imu_roll[imu_next_idx]-imu_roll[0], axes='szyx')
        #print(R_0i - R_0i_alt)
        imu_acc = np.array([imu_acc_x[imu_next_idx], imu_acc_y[imu_next_idx], imu_acc_z[imu_next_idx]])
        #imu_acc = np.array([0.25, 0, 0])
        imu_acc_corrected = np.dot(R_Ii, imu_acc)
        #imu_acc_corrected = imu_acc

        #print('sadfsadf')
        #print(imu_acc_corrected)
        #imu_acc_corrected = np.hstack(np.dot(R_LG, imu_acc_corrected))
        #print(imu_acc_corrected)
        acc_x = imu_acc_corrected[0]
        acc_y = imu_acc_corrected[1]
        acc_z = imu_acc_corrected[2]
        measurement = np.array([0, 0, 0, 0, 0, 0, acc_x, acc_y, acc_z]).T
        #measurement = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]).T
        #measurement = np.array([0, 0, 0, 0, 0, 0, imu_acc_x[imu_next_idx], imu_acc_y[imu_next_idx], imu_acc_z[imu_next_idx]]).T
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
        timestamp_next = imu_next

    time_diff = timestamp_next - timestamp
    A = state_transition(time_diff)
    model_cov_pos = (time_diff**3);
    model_cov_vel = (time_diff**2);
    model_cov_acc = time_diff;
    model_cov = np.block([
        [model_cov_pos * np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
        [np.zeros((3,3)), model_cov_vel * np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), np.zeros((3,3)), model_cov_acc * np.eye(3)],
        ])

    #print("measurement = {}".format(measurement))
    current_filtered_state_mean, current_filtered_state_covariance = kf.filter_update(
        current_filtered_state_mean,
        current_filtered_state_covariance,
        observation=measurement,
        transition_matrix=A,
        #transition_offset=np.zeros(9),
        transition_covariance=model_cov,
        observation_matrix=observation_matrix,
        observation_offset=np.array([0, 0, 0, 0, 0, 0, 0, 0, -g]), # TODO Is this right?
        observation_covariance=obs_cov,
    )

    if gps_next < imu_next:
        print("x = {}".format(current_filtered_state_mean))

    state_means = np.vstack((state_means, current_filtered_state_mean))
    state_covs = np.vstack((state_covs, [current_filtered_state_covariance]))
    #scatter.set_offsets(state_means[:,0:2])

    #ax.relim()
    #ax.autoscale_view()
    #plt.draw()
    #plt.pause(0.01)

    timestamp = timestamp_next


# TODO Use kalman smoother to have a more continuous state estimation

data_x = state_means[:,0]
data_y = state_means[:,1]
data_z = state_means[:,2]

max_vars = np.zeros((len(data_x),1))
for i in range(len(max_vars)):
    cov_mat = state_covs[i]
    max_vars[i] = np.mean(np.diag(cov_mat))
print(np.max(max_vars))
#scatter = ax.scatter(data_x, data_y, s=10, color='r')
scatter = ax.scatter(data_x, data_y, s=10, color='r')

plt.show()




















