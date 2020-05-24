import numpy as np
import cv2
import utility
import define_constants as const

if const.is_camera_feed:
    cap = cv2.VideoCapture(const.camera_num)
    cap.set(3, const.frame_width)
    cap.set(4, const.frame_height)
else:
    cap = cv2.VideoCapture(const.video_path)

# Initialize Trackbars
utility.initialize_trackbars()

while cap.isOpened():
    ret, frame = cap.read() # Read frames
    if not const.is_camera_feed:
        frame = cv2.resize(frame, (const.frame_width, const.frame_height), None)

    # Copy frame for canny_frame, point_wrap_frame and final_frame
    canny_frame = frame.copy()
    points_wrap_frame = frame.copy()
    final_frame = frame.copy()

    # Undistort frame
    undist_frame = utility.undistort(frame)

    # Preprocess frame
    edge_detected, frame_color_filtered, frame_preprocessed = utility.preprocess_frame(undist_frame)

    # Comparing filters
    if const.is_only_compare_filters:
        sobel_x_y_frame, sobel_x_frame, sobel_y_frame = utility.apply_sobel(frame)
        laplacian_frame = utility.apply_laplacian(frame)
        canny_frame = utility.apply_canny(frame)
    else:
        # Get Source and Destination points in frame
        src, dst = utility.get_src_dst()

        # Display src Points on frame
        frame_src_points = utility.display_points(points_wrap_frame, src)

        # Apply Prespective Wrap (Birdseye view)
        wrapped_frame = utility.perspective_warp(frame=frame_preprocessed, src=src, dst=dst, dst_frame_size=(const.frame_width, const.frame_height))

        # Display sliding windows
        frame_sliding_windows, nonzero_x, nonzero_y, left_lane_indices, right_lane_indices = utility.display_sliding_window(wrapped_frame)

        # Fit curve for left and right lane indices
        try:
            left_curve, right_curve = utility.get_left_right_curve(nonzero_x, nonzero_y, left_lane_indices, right_lane_indices)
        except Exception as e:
            print('Please place the points correctly\t___________________________')

        # Draw lanes
        try:
            frame_with_lanes = utility.display_lanes(frame, left_curve, right_curve, src=src, dst=dst)
        except Exception as e:
            print('Please place the points correctly\t---------------------------')

        # Fit left and right curve (with y values)
        try:
            left_fit, right_fit, y_max, meter_per_pixel_x, meter_per_pixel_y = utility.fit_curve_with_y_value(left_curve, right_curve)
        except Exception as e:
            print('Please place the points correctly\t############################')

        # Get radius of curvature
        radius_of_curv = utility.get_radius_of_curv(left_fit, right_fit, y_max, meter_per_pixel_y)

        # Get Offset
        offset = utility.get_offset(left_fit, right_fit, y_max, meter_per_pixel_x)

        # Display Lines
        frame_with_lines = utility.display_lines(frame_with_lanes, offset * const.move_line_offset_units)

        # Display Steering wheel
        steering_wheel_image = cv2.imread(const.steering_wheel_path)
        rotated_steering_wheel = utility.rotate_steering_wheel(steering_wheel_image, offset * const.power_steering_units)

        # Display text on frame
        frame_with_text = utility.display_text(frame_with_lines, radius_of_curv, offset)

    try:
        # Read the plotted histogram
        histogram_plot = cv2.imread('assets/plots/histogram_plot.jpg')
        # Combine with frame_with_lanes
        frame_with_histogram = cv2.addWeighted(frame, 0.7, histogram_plot, 0.4, 0)
    except Exception as e:
        print('Exception in reading histogram plot...')

    # Visualizing
    if const.is_only_compare_filters:
        cv2.imshow('Sobel x', sobel_x_frame)
        cv2.imshow('Sobel y', sobel_y_frame)
        cv2.imshow('Sobel x and y combined', sobel_x_y_frame)
        cv2.imshow('Laplacian Filter', laplacian_frame)
        cv2.imshow('Canny Filter', canny_frame)
    else:
        if const.is_demo:
            cv2.imshow('Source points', frame_src_points)
            cv2.imshow('Steering Wheel', rotated_steering_wheel)
            cv2.imshow(f"Lanes, edge_detector : {const.edge_detector}", frame_with_lanes)

            if const.is_show_histogram_plot:
                cv2.imshow('Histogram Plot', frame_with_histogram)
        else:
            cv2.imshow('Original Frame', frame)
            cv2.imshow(f"{const.edge_detector} Filter", edge_detected)
            cv2.imshow('Preprocessed Frame', frame_preprocessed)
            cv2.imshow('Birdseye', wrapped_frame)
            cv2.imshow('Sliding Windows', frame_sliding_windows)
            cv2.imshow('Source points', frame_src_points)
            cv2.imshow('Lanes', frame_with_text)

            if const.is_show_histogram_plot:
                cv2.imshow('Histogram Plot', frame_with_histogram)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
