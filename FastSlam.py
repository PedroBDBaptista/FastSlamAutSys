import numpy as np
import math
import random
import matplotlib.pyplot as plt
import cv2
import rosbag #Assuming we have rosbag installed


#def read_rosbag_data(filename):
   # bag=rosbag.Bag(filename)
    #Extract relevant data from the bag
    #These should be the measurements (aruco detections)
    #Return the extracted data as arrays or lists

def motion_model(particle):
    #Implement the motion model to predict next position
    #We will assume a constant velocity + noise approach
    global linear_vel
    global angular_vel

    #Define noise
    mu_xy=0
    sigma_xy=1
    mu_theta=0
    sigma_theta=1
    noise_x = np.random.normal(mu_xy,sigma_xy)
    noise_y = np.random.normal(mu_xy,sigma_xy)
    noise_theta=np.random.normal(mu_theta,sigma_theta)
    #Define the matrices for motion model
    matrixA=np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
    arrayB=np.array([linear_vel*dt*math.cos(particle['pose'][2]),
                     linear_vel*dt*math.sin(particle['pose'][2]),
                     angular_vel*dt])
    arrayNoise=np.array([noise_x,noise_y,noise_theta])
    #Update the new position
    pose_array=np.array(particle['pose'])#Turns the pose of the particle into an array for matrix multiplication
    new_pose= matrixA @ pose_array + arrayB + arrayNoise
    x,y,theta = new_pose
    particle['pose'] = [x,y,theta]
    #Return the new particle with the new 'pose'
    return particle

def h_inverse(particle,z):
    alpha=math.atan2(z[2],z[1])
    d=math.sqrt(z[1]**2 + z[2]**2)
    mu_x=particle['pose'][0] + d*math.cos(alpha + particle['pose'][2] - math.pi/2)
    mu_y=particle['pose'][1] + d*math.sin(alpha + particle['pose'][2] - math.pi/2)
    return mu_x, mu_y

def h_function(x,y,theta,mu_x,mu_y):
    dx=mu_x-x
    dy=mu_y-y
    d=math.sqrt(dx**2+dy**2)
    z_x=d*math.cos(math.pi/2 - theta + math.atan2(dy,dx))
    z_y=d*math.sin(math.pi/2 - theta + math.atan2(dy,dx))
    return z_x, z_y

def jacobian(x,y,theta,mu_x,mu_y):
    #Define elements to go inside the matrix -> function values and small deviations
    z_x,z_y=h_function(x,y,theta,mu_x,mu_y)
    x_x,y_x=h_function(x+precision,y,theta,mu_x,mu_y)
    x_y,y_y=h_function(x,y+precision,theta,mu_x,mu_y)

    #Define matrix
    H=np.array([(x_x-z_x)/precision,(x_y-z_x)/precision],
               [(y_x-z_y)/precision,(y_y-z_y)/precision])
    
    return H

def measurement_model(z,x):
    #Calculates the expected measurement given the current estimate of the landmark
    #Return the expected measurement to be used in EKF update probably
    return()

    
def is_landmark_seen(particle, landmark_id):
    landmarks = particle['landmarks']
    if landmarks:
        return any(landmark['id'] == landmark_id for landmark in landmarks)
    else:
        return False


def initialize_landmark(particle,z,err,landmark_id):
    mu=h_inverse(particle,z)
    H=jacobian(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu[0],mu[1])
    Q=np.eye(2)*err
    sigma= np.linalg.inv(H) @ Q @np.transpose(np.linalg.inv(H))
    weight=base_weight
    new_landmark={
        'id':landmark_id,
        'mu':mu,
        'sigma':sigma,
     }
    return new_landmark
    
def update_landmark(particle, landmark_id, z, err):
     # Implements the EKF update for a single landmark
    # mu: Mean of the landmark estimate
    # sigma: Covariance matrix of the landmark estimate
    # z: Measurement of the landmark (landmar_id,landmark_x,landmark_y)
    # Q: Measurement noise covariance matrix

    #We just want to alter the landmark whose id matches the landmark_id
    landmarks=particle['landmarks']
    for i,landmark in enumerate(landmarks):
        if landmark['id']==landmark_id:
            index=i
            break
    
    mu_old=particle['landmarks'][index]['mu']
    sigma_old=particle['landmarks'][index]['sigma']
    z_hat=h_function(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu_old[0],mu_old[1])
    H = jacobian(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu_old[0],mu_old[1])
    Q=H @ sigma_old @ np.transpose(H) + np.eye(2)*err
    K= sigma_old @ np.transpose(H) @ np.linalg.inv(Q)

    #Create array (our z contains (id,landmark_x, landmark_y) whereas our z_hat contains (landmark_x,landmark_y))
    z_deviation=np.array([z[1]-z_hat[0],z[2]-z_hat[1]])
    mu_new = mu_old + K @ z_deviation
    sigma_new= (np.eye(2) - K@H) @ sigma_old
    Qdet=np.linalg.det(2*np.pi*Q)
    new_weight=Qdet**(-1/2)*np.exp(-1/2*np.transpose(z_deviation) @ np.linalg.inv(Q) @ z_deviation)

    #Apply the new values to the respective landmark and the new weight to the particle
    particle['weight'] = new_weight
    particle['landmarks'][index]['mu']=mu_new
    particle['landmarks'][index]['sigma']=sigma_new
    return(particle)

def resample_particles(ParticleSet,num_particles):
    #THE WEIGHTS IN THE PARTICLES SHOULD BE NORMALIZED
    weights=[]
    for i in range(num_particles):
        weights.append(ParticleSet[i]['weight'])
    indices=np.arange(num_particles)
    resampled_indices=np.random.choice(indices,size=num_particles,p=weights)
    new_particles=[]
    new_particles=[ParticleSet[i] for i in resampled_indices]
    return (new_particles)

def fastslam_kc(ParticleSet,num_particles,measurements):
    for k in range(num_particles):
        #Sample new pose -> Motion Model
        ParticleSet[k]=motion_model(ParticleSet[k])
        #Loop in the number of observations done in each instant 
        #(there might be a possibility that the robot does multiple observations at the same instant)
        for i in range(measurements):
            landmark_id=measurements[i][0]
            #See if landmark as been seen
            if not is_landmark_seen(ParticleSet[k],landmark_id):
                new_landmark=initialize_landmark(ParticleSet[k],measurements[i],err,landmark_id)
                ParticleSet[k]['landmarks'].append(new_landmark)
                ParticleSet[k]['weight'] = base_weight
            else:
                ParticleSet[k]=update_landmark(ParticleSet[k],landmark_id,measurements[i],err)

    
    #Normalize the weights
    
    return ParticleSet #for each t

#Some parameters to define, such as timestep, linear_vel and angular_vel
time=0
dt=0.1 #(s)
linear_vel=0.5 #(m/s)
angular_vel=0.5 #(rad/s)
precision=0.001
err=0.5
#Load data from rosbag
#filename='path_to_rosbag_file'
#bag = read_rosbag_data(filename)

#Define the range for each dimension
x_min=0
x_max=20
y_min=0
y_max=20
theta_min=0
theta_max= 2*np.pi


#Initiate the ParticleSet:
num_particles=100
base_weight=1/num_particles
num_landmarks=5 #Put here the number of the landmarks. We should know their id and it should be by order.
particle_set=[] #Holds each particle. Each particle is a dictionary that should have 'pose' and 'landmarks'.
                #The 'pose' section has a list of 3 variables (x,y,theta)
                #The landmarks section has, for each landmark, a list for the 'mu' and a matrix 'sigma'

#We assume a random uniform distribution for the robot's pose in the particles. 
#We don't initialize mean values nor covariances for the landmarks because the robot has not yet detected them
for i in range(num_particles):
    x=random.uniform(x_min,x_max)
    y=random.uniform(y_min,y_max)
    theta=random.uniform(theta_min,theta_max)
    new_particle={
        'pose': [x,y,theta],
        'landmarks': [],
        'weight': 1/num_particles
    }
    
    #Loop for each landmark
    for j in range(num_landmarks):
        new_landmark={
            'id': [], #Assuming we use the ids in order, i.e, if we use 5 markers, we are using those which have id=0,1,2,3,4
            'mu': [],
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
            translation_y = fiducial.transform.translation.z

            #Add the landmark measurements to a variable. In this case we are not discarding the possibility of the robot detecting more than one aruco marker
            measurements.append([fiducial_id,translation_x,translation_y])


    particle_set, robot_pose, landmark_pose = fastslam_kc(particle_set, num_particles, measurements)

#Something to show the professor because he is an imbecile
#Create a video of the particle's positions
#video_filename='graph_evolution.mp4'
#fourcc=cv2.VideoWriter_fourcc(*'mp4v')
#video=cv2.VideoWriter(video_filename,fourcc,10,(800,800))
#Extract particle's positions
#x=[particle['pose'][0] for particle in particle_set]
#y=[particle['pose'][1] for particle in particle_set]

#Plot the particle's positions
#plt.scatter(x,y)
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Particles Positions (no resampling nor observations) t={:.1f}'.format(time))
#plt.grid(True)

#plt.savefig('particle_positions.png')
#plt.clf()

#Read the saved image using OpenCV
#frame=cv2.imread('particle_positions.png')
#frame=cv2.resize(frame,(800,800))
#video.write(frame)
#for i in range(100):
#    time=time+dt
#    particle_set=fastslam_kc(particle_set,num_particles)

    #Extract particle's positions
#    x=[particle['pose'][0] for particle in particle_set]
#    y=[particle['pose'][1] for particle in particle_set]

    #Plot the particle's positions
#    plt.scatter(x,y)
#    plt.xlabel('X')
#    plt.ylabel('Y')
#    plt.title('Particles Positions (no resampling nor observations) t={:.1f}'.format(time))
#    plt.grid(True)

#    plt.savefig('particle_positions.png')
#    plt.clf()
    #Read the saved image using OpenCV
#    frame=cv2.imread('particle_positions.png')
#    frame=cv2.resize(frame,(800,800))
#    video.write(frame)

#video.release()
#cv2.destroyAllWindows()