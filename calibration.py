'''
    File name: calibration.py
    Author: Victor Lyra
    Date created: 17/02/2020
    Date last modified: 20/25/2013
    Python Version: 3.7
'''

import numpy as np
import cv2
import glob, os

# VARIABLES
frames_from_video = True # If true => Extract frames from a video file
undistort = True         # If true => Undistort all calibration images
save_new_matrix = True   # Saves new camera matrix after undistortion
camera_name = 'robot_camera'    # Name directories, video and output file
extension = 'avi'        # Video Extension
video = ''               # If empty => videos/{camera_name}.mp4
skip_frames = 10         # Number of frames to skip in video
board_size = (9, 7)      # Chess Board Ratio
square_size = 0.02       # Board's square size in meters


def video_to_frames(skip_frames, video):
    print('Extracting frames from video...')
    directory = 'images/' + camera_name
    try:
        os.mkdir(directory)
    except os.error:
        print('{} folder already exists.'.format(directory))

    if video == '':
        video = 'videos/{}.{}'.format(camera_name, extension)
    video = cv2.VideoCapture(video)

    ret, frame = video.read()
    num_frame = 0
    image_name = 0

    while ret:
        if(num_frame % skip_frames == 0):
            cv2.imwrite(directory + '/{:03d}.jpg'.format(image_name), frame)
            image_name += 1

        num_frame += 1
        ret, frame = video.read()

def write_matrix(out_file, matrix):
    for i in range(3):
        for j in range(3):
            out_file.write('{0:.6f} '.format(matrix[i, j]))
        out_file.write('\n')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

if frames_from_video:
    video_to_frames(skip_frames, video)

images = glob.glob('images/{}/*'.format(camera_name))
images.sort()

num_found = 0
for fname in images:
    img = cv2.imread(fname)
    # img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    gray = img[:, :, 1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        num_found += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        print("Num patterns found: {}/{}".format(num_found, len(images)))
        img = cv2.drawChessboardCorners(img, board_size, corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(5)

cv2.destroyAllWindows()

print('Getting Calibration Values...')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Calibration file
print('Writing Calibration File...')
distortion = ''
for k in dist[0]:
    distortion += '{0:.5f}'.format(k) + ' '
with open('calibration/' + camera_name + '.txt', 'w') as calibfile:
    calibfile.write('Intrinsic Matrix:\n')
    write_matrix(calibfile, mtx)
    calibfile.write('\nDistortion Coefficients (k1 k2 p1 p2 k3):\n')
    calibfile.write(distortion)

    if undistort:
        print('Undistorting Images...')
        directory = 'undistorted/{}'.format(camera_name)
        try:
            os.mkdir(directory)
        except os.error:
            print('{} folder already exists.'.format(directory))

        for i, fname in enumerate(images):
            img = cv2.imread(fname)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            if save_new_matrix:
                calibfile.write('\n\nNew Camera Matrix (after undistortion):\n')
                write_matrix(calibfile, newcameramtx)
                save_new_matrix = False

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            cv2.imwrite(directory + '/undist{:03d}.png'.format(i), dst)


# Save with OpenCV file storage
cv_file = cv2.FileStorage('calibration/' + camera_name + '.yaml', cv2.FILE_STORAGE_WRITE)
cv_file.write("K", mtx)
cv_file.write("D", dist)
cv_file.write("newK", newcameramtx)
# note you *release* you don't close() a FileStorage object
cv_file.release() 
