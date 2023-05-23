import numpy as np
import random
import matplotlib.pyplot as plt
import rosbag # Assuming we have rosbag installed

def read_rosbag_data(filename):
    bag=rosbag.Bag(filename)
    #Extract relevant data from the bag
    #These should be the measurements (aruco detections)
    #Return the extracted data as arrays or lists

def motion_model(x,u):
    #Implement the motion model to predict next position
    #Probably we don't need controls "u" because we do not have odometry -> assume constant velocity?
    #Return the new pose (x_next)
    return x_next

def measurement_model(z,x):
    #Calculates the expected measurement given the current estimate of the landmark
    #Return the expected measurement to be used in EKF update probably

    
def is_landmark_seen(particle, landmark_id):
    for landmark in particle['landmarks']:
        if landmark['id'] == landmark_id:
            return True  # Landmark already seen in the particle
    return False  # Landmark is new

def ekf_update_landmark(mu, sigma, z, Q):
    # Implements the EKF update for a single landmark
    # mu: Mean of the landmark estimate
    # sigma: Covariance matrix of the landmark estimate
    # z: Measurement of the landmark
    # Q: Measurement noise covariance matrix

def resample_particles(particles, weights):
    #return the resampled particles

def fastslam_kc(ParticleSet,measurements):
    return ParticleSet,pose,landmarks #for each t


#Load data from rosbag
filename='path_to_rosbag_file'
bag = read_rosbag_data(filename)

#Define the range for each dimension
x_min=0
x_max=20
y_min=0
y_max=20
theta_min=-np.pi
theta_max= np.pi


#Initiate the ParticleSet:
num_particles=100
num_landmarks=5 #Put here the number of the landmarks. We should know their id and it should be by order.
particle_set=[] #Holds each particle. Each particle is a dictionary that should have 'pose' and 'landmarks'.
                #The 'pose' section has a list of 3 variables (x,y,theta)
                #The landmarks section has, for each landmark, a list for the 'mu' and a matrix 'sigma'

#We assume a random uniform distribution for the robot's pose in the particles. 
#We don't initialize mean values nor covariances for the landmarks because the robot has not yet detected them
for i in range(num_particles):
    x=random.uniform(x_min,x_max)
    y=random.uniform(y_min,y_max)
    theta=random.uniform((theta_min,theta_max))
    new_particle={
        'pose': [x,y,theta]
        'landmarks': []
    }
    
    #Loop for each landmark
    for j in range(num_landmarks):
        new_landmark={
            'id': j #Assuming we use the ids in order, i.e, if we use 5 markers, we are using those which have id=0,1,2,3,4
            'mu': []
            'sigma': []
        }
        new_particle['landmarks'].append(new_landmark)
    
    #Add the new_particle to the particle_set variable
    particle_set.append(new_particle)
    


#Iterate over the messages in the bag file
for topic, msg, t in bag.read_messages():
    measurements=[]
    if topic == '/fiducial_transforms':
        for fiducial in msg.fiducials:
            fiducial_id = fiducial.fiducial_id
            translation_x = fiducial.transform.translation.x
            translation_y = fiducial.transform.translation.y
            #translation_z = fiducial.transform.translation.z

            #Add the landmark measurements to a variable. In this case we are not discarding the possibility of the robot detecting more than one aruco marker
            measurements.append([fiducial_id,translation_x,translation_y])


    particle_set, robot_pose, landmark_pose = fastslam_kc(particle_set, num_particles, measurements)