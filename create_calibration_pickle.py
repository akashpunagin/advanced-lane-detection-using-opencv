import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob #  glob module is used to retrieve files/pathnames matching a specified pattern
import pickle
import define_constants as const
# plt.tight_layout()

# Read all Distorted images
images = glob.glob(const.calibration_images_path)

# Camera Calibration

# Store chessboard coordinates
chess_points = [] #these will all be the same since it's the same board
# To store corners of chessboard
corners = []

# Chess board is 6 rows by 9 columns (on the outermost edge)
# Generate list of (x,y,z) coordinates for each combination, (z will always by 0)
chess_point = np.zeros((9*6, 3), np.float32)

# z stays zero. set xy to grid values
chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Iterate through each image, find corners and append to corners
print('Finding corners...\n')
for i, image in enumerate(images):
    # Read individual image
    img = mpimg.imread(image)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find corners in the image, returns boolean and coordinates of corners
    success, corner_coordinates = cv2.findChessboardCorners(image=gray, patternSize=(9,6), corners=None) # corners – Output array of detected corners

    if success:
        print(f'For image {i+1} : Found')
        corners.append(corner_coordinates)
        chess_points.append(chess_point)
    else:
        print(f'For image {i+1} : Not Found. File - {image}')

# Visualize Corners in image
print('\nDisplaying images with corners...\n')
for n_img in range(6,12):
    # image = r'assets/images/calibration_images/calibration' + str(n_img) + r'.jpg'
    image = mpimg.imread(images[n_img])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    ax1.imshow(image);
    ax1.set_title('Captured Image', fontsize=8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    success, corner_coordinates = cv2.findChessboardCorners(image=gray, patternSize=(9,6), corners=None)
    if success == False:
        print(f'Corners not found for image - {n_img}')
        continue
    corners_drawn_img = cv2.drawChessboardCorners(image=image,patternSize=(9,6),corners=corner_coordinates,patternWasFound=success)

    ax2.imshow(corners_drawn_img);
    ax2.set_title('Corners drawn Image', fontsize=8)

    # plt.savefig('saved_figures/chess_corners.png')

# Show chessboards with corners
# plt.show()

# Distortion Correction

# Distortion Coefficients - (k1 k2 p1 p2 k3) ie (ret, mtx, dist, rvecs, tvecs)
# By these values, camera can be calibrated and images can be undistorted
# mtx: Camera Matrix, which helps to transform 3D objects points to 2D image points
# dist: distortion coefficient
# rvecs - rotation vectors
# tvecs- translation vectors
# rvecs and tvecs - position of camera in real world

img_shape = mpimg.imread(images[0]).shape
print('Image shape : ', img_shape)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                            objectPoints=chess_points,
                            imagePoints=corners,
                            imageSize=(img_shape[1], img_shape[0]), cameraMatrix=None, distCoeffs=None)

# Save pickle file
camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
# camera["imagesize"] = img_size
pickle.dump(camera, open(const.pickle_save_path, "wb"))
print('\nPickle file created...')
