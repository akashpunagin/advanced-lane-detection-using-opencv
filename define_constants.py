import cv2

# VISUALIZING
is_demo = True
is_show_histogram_plot = False
is_only_compare_filters = False

# EDGE DETECTION FILTERS (SELECT ONE OUT OF THREE)
is_apply_canny = True
is_apply_sobel = False
is_apply_laplacian = False

if is_apply_canny: edge_detector = "Canny"
elif is_apply_sobel: edge_detector = "Sobel"
elif is_apply_laplacian : edge_detector = "Laplacian"

# VIDEO SETUP
video_path = r'assets/video/test_video.mp4'
# video_path = r'assets/video/test_video_challenge.mp4'
# video_path = r'assets/video/test_video_solidWhiteRight.mp4'
# video_path = r'assets/video/test_video_solidYellowLeft.mp4'
# video_path = r'assets/video/test_video_harder_challenge.mp4'

# CAMERA SETUP
is_camera_feed = False
camera_num = 0
frame_width = 640
frame_height = 480

if is_camera_feed:
    initial_trackbar_val = [24,55,12,100] #width-top,height-top,width-bottom,height-bottom
else:
    initial_trackbar_val = [42,63,14,88] #width-top,height-top,width-bottom,height-bottom
    # initial_trackbar_val = [30, 70,14,88] # ONLY FOR HARDER CHALLENGE TEST VIDEO

# SELECT PATH FOR PICKLE FILE
# pickle_dir = r'assets/pickle_file/created_pickle_cal.p'
pickle_dir = r'assets/pickle_file/saved_pickle_cal.p'

# SLIDING WINDOW PARAMETERS
n_windows = 15
margin = 50
min_pixel = 1
window_color = (100, 255, 255)
window_thickness = 1
pixel_color_in_window_left = (100, 255, 255)
pixel_color_in_window_right = (20, 20, 255)

# LANE
lane_color = (0, 200, 255)

# TEXT
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 1
direction_font_color = (0, 160, 200)
radius_offset_font_color = (255, 255, 0)

# STEERING WHEEL
steering_wheel_path = r'assets/images/steering_wheel_300.png'
power_steering_units = 7

# LINES
move_line_offset_units = 2000
lines_color = (0,0, 255)
mid_line_color = (0, 255, 0)
offset_line_color = (255, 0, 0)

# IMAGE CALIBRATION (Generate picke file)
calibration_images_path = r'assets/images/calibration_images/calibration*.jpg'
pickle_save_path = r'assets/pickle_file/created_pickle_cal.p'
