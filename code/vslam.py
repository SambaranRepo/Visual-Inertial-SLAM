from xxlimited import new
import numpy as np
import numpy.linalg as la
from pr3_utils import *
from scipy.linalg import expm,block_diag
from pr3_utils import *
from tqdm import tqdm
from glob import glob

def projection_jacobian(q, Nt): 
	'''
	: Given a vector q of size R^{4}, and the projection function pi = 1/q_3 * q, return the jacobian matrix that is the differentiation of pi wrt q
	: Since q is of size R^{4} and pi is of size R^{4,4}, jacobian is of shape R^{4,4}. 
	: We have the q vector for Nt different landmarks and hence we will have a jacobian for each of these different landmarks. 
	: Jacobian will be of shape 4*4*Nt
	'''
	dpi_dq = np.zeros((4,4,Nt))
	dpi_dq[0,0,:] = 1/q[2,:]
	dpi_dq[1,1,:] = 1/q[2,:]
	dpi_dq[1,1,:] = 1/q[2,:]
	dpi_dq[3,3,:] = 1/q[2,:]
	dpi_dq[0,2,:] = -q[0,:]/np.square(q[2,:])
	dpi_dq[1,2,:] = -q[1,:]/np.square(q[2,:])
	dpi_dq[3,2,:] = -1/np.square(q[2,:])
	return dpi_dq

def get_flat_indices(true_columns,size_M):
	'''
	: Given that landmark is detected at certain columns, we want to find out what will be the location of the landmarks coordinates when it is flattened
	: Assign True to columns at positions where landmarks are detected
	: Reshape this 3,M matrix to 3M matrix by first taking transpose to convert to M,3 matrix and then reshaping into 3M matrix using reshape command
	: By default, reshape will take elements along the column first because of C-type indexing format. 
	: Hence we will have landmark x,y,z followed by next landmark x,y,z and so on. 
	: This function returns positions in the 1D array that has locations for landmarks which are indicated by True value in the cells
	: This is useful when we want to transform 2D arrays that contain landmark values to a single dimensional array that is then used to do the EKF update step.
	'''
	
	boolean_array = np.full((3,size_M),False)
	boolean_array[:,true_columns] = True
	return np.where( (boolean_array.T).reshape((1,3*size_M)) )[1]

def curly_hat(v,w):
	'''
	: Curly hat operator used to define the perturbation kinematics and hence used to get the covariance of the IMU pose in the predict step
	'''
	w_hat, v_hat = hat(w), hat(v)
	u_curly = np.block([[w_hat, v_hat], [np.zeros((3,3)), w_hat]])
	return u_curly 

def hat(x):
	'''
	: Given a vector x, return the skew symmetric se(3) matrix of the vector
	: Used to give the skew symmetric matrix of the angular velocity for axis-angle notation and pose SE(3) kinematics
	'''
	return np.asarray([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def twist(x): 
	'''
	: Get the hat map of the 6*1 vector which defines the twist matrix
	'''
	theta = x[-3:]
	p = x[:3]
	twist = np.block([[hat(theta), p], [np.zeros((1,4))]])
	return twist

def circular_dot(s): 
	'''
	: Get the circular dot operator of the matrix
	: Used to change the cross product order of delta_mu(hat) and (mu^(-1) @ m_j)
	: Used in update step of IMU pose using the EKF
	'''
	circled = np.zeros((4,6,s.shape[-1]))
	circled[0,0,:] = 1
	circled[1,1,:] = 1
	circled[2,2,:] = 1
	circled[0,4,:] = s[2,:]
	circled[0,5,:] = -s[1,:]
	circled[1,3,:] = -s[2,:]
	circled[1,5,:] = s[0,:]
	circled[2,3,:] = s[1,:]
	circled[2,4,:] = -s[0,:]
	return circled

def IMU_localization(T_initial,linear_velocity, angular_velocity, tau):
	'''
    : Given the control inputs linear and angular velocity and the previous pose, returns the current next pose of the robot
    '''
	omega_hat = hat(angular_velocity)
	twist = np.block([[omega_hat, linear_velocity.reshape(3,-1)], [np.zeros((1,4))]])
	T_next = T_initial @ expm(tau * twist)
	return T_next

if __name__ == '__main__':

    # Load the measurements
    filename = glob("code/data/03.npz")[0]
    t,features1,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    #stereo camera matrix
    Ks = np.asarray([[K[0][0], K[0][1], K[0][2], 0], [K[1][0], K[1][1], K[1][2], 0], [K[0][0],K[0][1],K[0][2],-K[0][0] * b], [K[1][1],K[1][1],K[1][2],0]])
    P = np.asarray([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    features = features1[:,::6,:]
    M = features.shape[1]
    map = np.full((3,M), np.NaN)
    mu_map_t = np.zeros((3,M))
    mu_map_t_flat = np.zeros((3*M,1))
    sigma_map = np.eye(3*M) *0.1
    W = 1e-1 * np.eye(6)
    W[1,1] = W[2,2] = 1e-4
    W[3:,3:] = np.eye(3) * 5e-2
    sigma_IMU = np.eye(6) * 0.1
    V = 15
    pose_trajectory = []
    T = np.eye(4)
    #Initialise landmarks using first time features 

    index = np.sum(features[:,:,0], axis = 0) != -4
    new_points = np.isnan(np.sum(map, axis = 0))
    new_points = np.where(np.logical_and(index , new_points))[0]
    u_L = features[0,new_points,0]
    v_L = features[1,new_points,0]
    u_R = features[2,new_points,0]

	# Get (x,y,z) in optical frame using Stereo Camera Model
    x_o = (u_L - Ks[0,2])/Ks[0,0]
    y_o = (v_L - Ks[1,2])/Ks[1,1]
    z_o = Ks[2,3]/(u_R-u_L)
    xyz_o = z_o * np.vstack([x_o,y_o,np.ones(x_o.size),1/z_o])
    print(f'new points observed : {new_points}')
    print(f'prev coordinates at new points : {mu_map_t[:,new_points]}')
    map[:,new_points] = (T @ imu_T_cam @ xyz_o)[:3,:]
    observed_new_points_flat = get_flat_indices(new_points,M)
    N = new_points.size
    mu_map_t_flat[observed_new_points_flat,:] =  (map.T).reshape((3*M,1))[observed_new_points_flat,:]
    mu_map_t = mu_map_t_flat.reshape(3,M,order='F')
    print(f'new coordinates at new points : {mu_map_t[:,new_points]}')
    P = np.block([np.eye(3), np.zeros((3,1))])

    for i in tqdm(range(1, len(t[0]))):
        #a. IMU Predict step
        tau = t[0][i] - t[0][i-1]
        v = linear_velocity[:,i]
        w = angular_velocity[:,i]
        pose_trajectory.append(T)
        T = IMU_localization(T, v, w, tau)
        sigma_IMU = expm(-tau * curly_hat(v,w)) @ sigma_IMU @ expm(-tau * curly_hat(v,w)).T + W
        
        
        #b. Landmark EKF update step(independent of IMU update)
        index = np.sum(features[:,:,i], axis = 0) != -4
        unobserved_points = np.isnan(np.sum(map, axis = 0))
        observed_points = ~unobserved_points
        new_points = np.where(np.logical_and(index, unobserved_points))[0]
        update_points = np.where(np.logical_and(index , observed_points))[0]

        if new_points.size != 0: 
            u_L = features[0,new_points,i]
            v_L = features[1,new_points,i]
            u_R = features[2,new_points,i]
            x_o = (u_L - Ks[0,2])/Ks[0,0]
            y_o = (v_L - Ks[1,2])/Ks[1,1]
            z_o = Ks[2,3]/(u_R-u_L)
            xyz_o = z_o * np.vstack([x_o,y_o,np.ones(x_o.size),1/z_o])
            map[:,new_points] = (T @ imu_T_cam @ xyz_o)[:3,:]
            observed_new_points_flat = get_flat_indices(new_points,M)
            N = new_points.size
            mu_map_t_flat[observed_new_points_flat,:] =  (map.T).reshape((3*M,1))[observed_new_points_flat,:]
            mu_map_t = mu_map_t_flat.reshape(3,M,order='F')
        if update_points.size != 0: 
            update_points_flat = get_flat_indices(update_points, M)
            Nt = update_points.size
            z_t = features[:,update_points, i].reshape(-1,1, order = 'F')
            mu_bar = np.vstack((mu_map_t[:, update_points], np.ones(Nt)))
            q_opt = la.inv(imu_T_cam) @ la.inv(T) @ mu_bar
            z_pred = (Ks @ q_opt/q_opt[2,:]).reshape(-1,1, order = 'F')
            matrix1 = la.inv(imu_T_cam) @ la.inv(T) @ P.T 
            dpi_dq = projection_jacobian(q_opt, Nt)
            matrix2 = np.einsum('ijk,jl->ilk',dpi_dq,matrix1)
            Ht_o = np.einsum('ij,jkl->ikl',Ks,matrix2) #shape : 4 * 3 * Nt
            Ht_obs = block_diag(*Ht_o.T).T


            mu_bar = np.vstack((mu_map_t[:, update_points], np.ones(Nt)))
            q_opt = la.inv(imu_T_cam) @ la.inv(T) @ mu_bar
            z_pred = (Ks @ q_opt/q_opt[2,:]).reshape(-1,1, order = 'F')
            q_opt = la.inv(imu_T_cam) @ la.inv(T) @ mu_bar
            z_pred = (Ks @ q_opt/q_opt[2,:]).reshape(-1,1, order = 'F')
            matrix1 = la.inv(imu_T_cam) @ la.inv(T) @ P.T 
            dpi_dq = projection_jacobian(q_opt, Nt)
            matrix2 = np.einsum('ijk,jl->ilk',dpi_dq,matrix1)
            Ht_o = np.einsum('ij,jkl->ikl',Ks,matrix2) #shape : 4 * 3 * Nt
            Ht_obs = block_diag(*Ht_o.T).T
            mu = mu_map_t_flat[update_points_flat, :]
            row = np.tile(update_points_flat,(3*Nt,1)).reshape(-1,1,order='F')
            column = np.tile(update_points_flat,(3*Nt,1)).reshape(-1,1)
            sigma_prev = sigma_map[row,column].reshape(3*Nt,3*Nt)
            Kt_obs =  sigma_prev @ Ht_obs.T @ la.inv(Ht_obs @ sigma_prev @ Ht_obs.T + np.eye(4*Nt) * V)
            mu += Kt_obs @ (z_t - z_pred)
            sigma_prev = (np.eye(3*Nt) - Kt_obs @ Ht_obs) @ sigma_prev @ (np.eye(3*Nt) - Kt_obs @ Ht_obs).T + Kt_obs @ (np.eye(4*Nt) * V) @ Kt_obs.T
            mu_map_t_flat[update_points_flat, :] = mu
            mu_map_t = mu_map_t_flat.reshape(3,M,order = 'F')
            sigma_map[row,column] = sigma_prev.reshape(-1,1)

    pose = np.asarray(pose_trajectory)
    visualize_trajectory_2d(pose,mu_map_t, show_ori = True)
    filename = filename.split('/')[2].split('.')[0]
    plt.savefig(f'code/SLAM_{filename}_Part_a,b.png', format = 'png', bbox_inches = 'tight')
    plt.show(block = True)

    plt.close()






