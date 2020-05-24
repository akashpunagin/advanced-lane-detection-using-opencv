import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import define_constants as const

# Initialize Trackbars
def initialize_trackbars(initial_trackbar_vals = const.initial_trackbar_val):
    window_title = "Trackbars"
    cv2.namedWindow(window_title) # Create window
    cv2.resizeWindow(window_title, 560, 440) # Resize window

    # cv2.CreateTrackbar(trackbarName, windowName, value, count, onChange)
    cv2.createTrackbar(f"Width Top[{initial_trackbar_vals[0]}]", window_title, initial_trackbar_vals[0],50, on_change)
    cv2.createTrackbar(f"Height Top[{initial_trackbar_vals[1]}]", window_title, initial_trackbar_vals[1], 100, on_change)
    cv2.createTrackbar(f"Width Bottom[{initial_trackbar_vals[2]}]", window_title, initial_trackbar_vals[2], 50, on_change)
    cv2.createTrackbar(f"Height Bottom[{initial_trackbar_vals[3]}]", window_title, initial_trackbar_vals[3], 100, on_change)

def on_change(event):
    pass


# Valuate Trackbars
def get_src_dst(initial_trackbar_vals = const.initial_trackbar_val):
    window_title = "Trackbars"
    widthTop = cv2.getTrackbarPos(f"Width Top[{initial_trackbar_vals[0]}]", window_title)
    heightTop = cv2.getTrackbarPos(f"Height Top[{initial_trackbar_vals[1]}]", window_title)
    widthBottom = cv2.getTrackbarPos(f"Width Bottom[{initial_trackbar_vals[2]}]", window_title)
    heightBottom = cv2.getTrackbarPos(f"Height Bottom[{initial_trackbar_vals[3]}]", window_title)

    # Set source points
    src = np.float32([(widthTop/100,heightTop/100), # top-left
                    (1-(widthTop/100), heightTop/100), # top-right
                    (widthBottom/100, heightBottom/100), # bottom-right
                    (1-(widthBottom/100), heightBottom/100) # bottom-left
                    ])
    # Set destination points
    dst = np.float32([
                    (0,0), # top-left
                    (1, 0), # top-right
                    (0,1), # bottom-right
                    (1,1) # bottom-left
                    ])
    return src, dst


# Undistort Frame
def undistort(img, pickle_dir=const.pickle_dir):
    with open(pickle_dir, mode='rb') as file:
        pickle_file = pickle.load(file)
    mtx = pickle_file['mtx']
    dist = pickle_file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# Frame Preprocessing
def filter_colors(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yellow_low = np.array([18,94,140])
    yellow_high = np.array([48,255,255])
    white_low = np.array([0, 0, 200])
    white_high = np.array([255, 255, 255])
    white_mask= cv2.inRange(hsv,white_low,white_high)
    yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
    combined = cv2.bitwise_or(white_mask,yellow_mask)
    return combined

def preprocess_frame(frame):
    if const.is_apply_canny:
        edge_detected = apply_canny(frame)
    elif const.is_apply_sobel:
        edge_detected, sobel_x, sobel_y = apply_sobel(frame)
    elif const.is_apply_laplacian:
        edge_detected = apply_laplacian(frame)
    else:
        print('Select an edge detection filter in define_constants.py file ...')
    kernel = np.ones((5,5))
    dialate = cv2.dilate(edge_detected,kernel,iterations=1)
    erode = cv2.erode(dialate,kernel,iterations=1)
    color_filtered = filter_colors(frame)
    combined = cv2.bitwise_or(color_filtered, erode)
    return edge_detected, color_filtered, combined

def apply_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 100)
    return canny

def apply_sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(src=gray,ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_x_y = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Convert back to uint8
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_x_y = cv2.convertScaleAbs(sobel_x_y)

    return sobel_x_y, sobel_x, sobel_y

def apply_laplacian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(src=blur, ddepth=cv2.CV_16S, ksize=3) # Apply Laplacian filter
    laplacian = cv2.convertScaleAbs(laplacian) # Convert back to uint8
    return laplacian

# Display src points on frame
def display_points(frame, points):
    frame_size = np.float32([(frame.shape[1],frame.shape[0])])

    # Scale points to shape of frame_size
    points = points * frame_size
    # Iterate through points and put circle
    for x in range(0,4):
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(frame, (int(points[x][0]),int(points[x][1])), 10, (0,0,255), cv2.FILLED)

    # Lines to form a quadrangle
    cv2.line(frame, pt1=(int(points[0][0]), int(points[0][1])), pt2=(int(points[1][0]), int(points[1][1])), color=(0,0,255), thickness=2)
    cv2.line(frame, pt1=(int(points[0][0]), int(points[0][1])), pt2=(int(points[2][0]), int(points[2][1])), color=(0,0,255), thickness=2)
    cv2.line(frame, pt1=(int(points[2][0]), int(points[2][1])), pt2=(int(points[3][0]), int(points[3][1])), color=(0,0,255), thickness=2)
    cv2.line(frame, pt1=(int(points[3][0]), int(points[3][1])), pt2=(int(points[1][0]), int(points[1][1])), color=(0,0,255), thickness=2)

    return frame

# Prespective Wrapping (Birdseye view)
def perspective_warp(frame, src, dst, dst_frame_size = (1280, 720)):
    frame_size = np.float32([(frame.shape[1],frame.shape[0])])

    # Scale src and dst points to shape of image size
    src = src * frame_size
    dst = dst * np.float32(dst_frame_size)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply perspective transformation to img, (dst_size parameter-size of output image)
    warped = cv2.warpPerspective(frame, M, dst_frame_size)

    return warped

# Sliding Windows
def display_sliding_window(frame, n_windows=const.n_windows, margin=const.margin, min_pixel=const.min_pixel, draw_windows=True):

    # Frame to output, stack arrays
    frame_output = np.dstack((frame, frame, frame)) * 255

    # Calculate histogram of lower half of frame
    histogram = np.sum( frame [frame.shape[0]//2 : ,:], axis=0)

    # Save histogram to display
    save_histogram(histogram)

    # Find peaks in histogram, in left and right halves
    midpoint = int(histogram.shape[0] / 2)
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows, ie imgheight / nwindows
    window_height = np.int(frame.shape[0] / n_windows)

    # Identify the indices of nonzero pixels in the img
    nonzero_indices = frame.nonzero() # Returns a tuple of indices with non-zero elements (y dimention indices, x dimention indices)
    nonzero_y = np.array(nonzero_indices[0])
    nonzero_x = np.array(nonzero_indices[1])

    # Set the current x position to be updated for each window during itreation
    left_x_current = left_peak
    right_x_current = right_peak

    # Create empty lists to receive left and right lane pixel indices (in window)
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for nwindow in range(n_windows):
        # Identify window boundaries - y dimention
        win_y_low = frame.shape[0] - ( (nwindow + 1) * window_height )
        win_y_high = frame.shape[0] - ( nwindow * window_height )
        # Identify window boundaries - x dimention, left and right
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        # Draw window as rectangle
        if draw_windows == True:
            cv2.rectangle(frame_output, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), const.window_color, const.window_thickness)
            cv2.rectangle(frame_output, (win_xright_low, win_y_low), (win_xright_high, win_y_high), const.window_color, const.window_thickness)

        # Identify the nonzero pixels within the window
        left_lane_indices_in_window = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        right_lane_indices_in_window = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_indices.append(left_lane_indices_in_window)
        right_lane_indices.append(right_lane_indices_in_window)

        # recenter next window on their mean position
        if len(left_lane_indices_in_window) > min_pixel:
            left_x_current = np.int(np.mean(nonzero_x[left_lane_indices_in_window]))
        if len(right_lane_indices_in_window) > min_pixel:
            right_x_current = np.int(np.mean(nonzero_x[right_lane_indices_in_window]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Color the nonzero pixles in window
    frame_output[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = const.pixel_color_in_window_left # left lane
    frame_output[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = const.pixel_color_in_window_right # right lane

    return frame_output, nonzero_x, nonzero_y, left_lane_indices, right_lane_indices

def save_histogram(histogram):
    plt.plot(histogram)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('assets/plots/histogram_plot.jpg')
    plt.clf()
    plt.cla()
    plt.close()

def get_left_right_curve(nonzero_x, nonzero_y, left_lane_indices, right_lane_indices):

    # Get left and right nonzero lane pixel indices in x and y dimention
    left_x_indices = nonzero_x[left_lane_indices]
    left_y_indices = nonzero_y[left_lane_indices]
    right_x_indices = nonzero_x[right_lane_indices]
    right_y_indices = nonzero_y[right_lane_indices]

    # Empty lists to store A B and C values
    left_A, left_B, left_C = [], [], []
    right_A, right_B, right_C = [], [], []

    if left_x_indices.size and right_x_indices.size and left_y_indices.size and right_y_indices.size:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y_indices, left_x_indices, 2)
        right_fit = np.polyfit(right_y_indices, right_x_indices, 2)

        left_A.append(left_fit[0])
        left_B.append(left_fit[1])
        left_C.append(left_fit[2])

        right_A.append(right_fit[0])
        right_B.append(right_fit[1])
        right_C.append(right_fit[2])

        # np.empty - create new array with random values to store final A B C values
        left_fit_final = np.empty(3)
        right_fit_final = np.empty(3)

        # Find mean of last (minimum) 10 values of A B and C (left and right)
        left_fit_final[0] = np.mean(left_A[-10:])
        left_fit_final[1] = np.mean(left_B[-10:])
        left_fit_final[2] = np.mean(left_C[-10:])

        right_fit_final[0] = np.mean(right_A[-10:])
        right_fit_final[1] = np.mean(right_B[-10:])
        right_fit_final[2] = np.mean(right_C[-10:])

        # Generate y values for plotting
        y_values = np.linspace(0, const.frame_height - 1, const.frame_height)

        # Calculate curves -  fit = Ay2 + By + C
        left_curve = left_fit_final[0] * ( y_values ** 2 ) + left_fit_final[1] * y_values + left_fit_final[2]
        right_curve = right_fit_final[0] * ( y_values ** 2 ) + right_fit_final[1] * y_values + right_fit_final[2]

        return left_curve, right_curve
    else:
        return (0,0)


def display_lanes(frame, left_curve, right_curve, src, dst):
    # Generate y values
    y_values = np.linspace(0, const.frame_height - 1, const.frame_height)

    # Original camera frame, (not Birdseye view)
    inv_perspective_frame = np.zeros_like(frame)

    # Find left and right points and stack them horizontally
    left_points = np.array([np.transpose(np.vstack([left_curve, y_values]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_curve, y_values])))])
    points = np.hstack((left_points, right_points))

    # Fill the points with lane color
    cv2.fillPoly(inv_perspective_frame, np.int_(points), const.lane_color)

    # Apply inverse perspective wrap
    inv_perspective_wrapped_frame = perspective_warp(inv_perspective_frame, dst_frame_size=(const.frame_width,const.frame_height), dst=src, src=dst)

    # Combine frame and inverse perspective wrapped frame
    frame_combined = cv2.addWeighted(frame, 0.5, inv_perspective_wrapped_frame, 0.3, 0)

    return frame_combined




# Fit left and right curve with y values
def fit_curve_with_y_value(left_curve, right_curve):
    # Generate y values
    y_values = np.linspace(0, const.frame_height - 1, const.frame_height)

    # Coversion rates for pixels to metric
    meter_per_pixel_y = 1 / const.frame_height # meters per pixel in y dimension
    meter_per_pixel_x = 0.1 / const.frame_height  # meters per pixel in x dimension

    # Fit new polynomials (in metric)
    left_fit = np.polyfit(y_values * meter_per_pixel_y, left_curve * meter_per_pixel_x, 2)
    right_fit = np.polyfit(y_values * meter_per_pixel_y, right_curve * meter_per_pixel_x, 2)

    # Measure radius of curvature at the maximum y value (at bottom of the image)
    y_max = np.max(y_values)

    return left_fit, right_fit, y_max, meter_per_pixel_x, meter_per_pixel_y


# Calculate Radius of Curvature
def get_radius_of_curv(left_fit, right_fit, y_max, meter_per_pixel_y):
    # Calculate radius of curvature at y_max (in metric)
    # Formula : Rcurve = (( 1 + (2Ay + B )**2 )**(3/2)) / abs( 2A )
    radius_of_curv_left = ((1 + (2 * left_fit[0] * (y_max * meter_per_pixel_y) + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    radius_of_curv_right = ((1 + (2 * right_fit[0] * (y_max * meter_per_pixel_y) + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Average left and right radius of curvature
    radius_of_curv = round((radius_of_curv_left + radius_of_curv_right)/2)

    return radius_of_curv

# Calculate offset
def get_offset(left_fit, right_fit, y_max, meter_per_pixel_x):
    # Car position, width of frame / 2
    car_position = const.frame_width / 2

    # Calculate intercept - fit = Ay2 + By + C
    x_left_intercept = left_fit[0] * const.frame_height ** 2 + left_fit[1] * const.frame_height + left_fit[2]
    x_right_intercept = right_fit[0] * const.frame_height ** 2 + right_fit[1] * const.frame_height + right_fit[2]

    # Calculate offset by averaging left and right curve
    offset = (x_left_intercept + x_right_intercept) / 2

    # Convert center to metrics
    offset_in_meters = (car_position - offset) * meter_per_pixel_x

    return offset_in_meters

def display_lines(frame, offset):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    for x in range(-30, 30):
        w = frame_width // 20
        cv2.line(frame, (w * x + int(offset // 100), frame_height - 30),
                 (w * x + int(offset // 100), frame_height), const.lines_color, 2)
    cv2.line(frame, (int(offset // 100) + frame_width // 2, frame_height - 30),
             (int(offset // 100) + frame_width // 2, frame_height), const.mid_line_color, 3)
    cv2.line(frame, (frame_width // 2, frame_height - 50), (frame_width // 2, frame_height), const.offset_line_color, 2)

    return frame



def rotate_steering_wheel(image, offset):
    # Fine center of image and rotate
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center=image_center, angle=offset, scale=1.0)
    rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Set background to white
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    mask = cv2.compare(gray,5,cv2.CMP_LT) # Performs the per-element comparison of two arrays or an array and scalar value, CMP_LT - src1 is less than src2.
    rotated[mask > 0] = (255,255,255)

    return rotated

def display_text(frame, radius_of_curv, offset):
    if offset > 0.4:
        direction = 'Go Right'
    elif offset < -0.4:
        direction = 'Go Left'
    elif offset < 0.4 and offset > -0.4:
        direction = 'Go Straight'
    else:
        direction = 'No lane Found'

    # cv2.putText(image, text, org, font,fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Radius : {str(radius_of_curv)} m", ((const.frame_width//2)-100, 40), const.font, 0.7, const.radius_offset_font_color, const.thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Offset : {str(round(offset, 2))} m", ((const.frame_width//2)-100, 70), const.font, 0.7, const.radius_offset_font_color, const.thickness, cv2.LINE_AA)
    cv2.putText(frame, direction, ((const.frame_width//2)-100, 100), const.font, 0.7, const.direction_font_color, const.thickness, cv2.LINE_AA)

    return frame
