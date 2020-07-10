# Lane Detection using OpenCV
* This project will detect lanes in recorded video or via webcam.
* All the test videos are available in *assets/video* directory.
* Python version used - 3.6.10


## Required Python Packages

- OpenCV
- NumPy
- Matplotlib
- glob
- pickle

### How to Run:

1. Run create_calibration_pikle.py file to create calibration pickle file.
2. Edit define_constants.py file according to your needs. **If required**
3. Run lane_detection.py file.

## Output
![Canny Filter](README_media/Canny_Filter_screenshot.png "Canny Filter")
<br />
*Canny Filter*
<br /><br />

![Preprocessed Frame](README_media/Preprocessed_Frame_screenshot.png "Preprocessed Frame")
<br />
*Preprocessed Frame*
<br /><br />

![Birdseye](README_media/Birdseye_screenshot.png "Birdseye")
<br />
*Birdseye*
<br /><br />

![Histogram Plot](README_media/Histogram_Plot_screenshot.png "Histogram Plot")
<br />
*Histogram Plot on Frame*
<br /><br />

![Sliding Windows](README_media/Sliding_Windows_screenshot.png "Sliding Windows")
<br />
*Sliding Windows*
<br /><br />

![Source Points](README_media/Source_points_screenshot.png "Source Points")
<br />
*Source Points*
<br /><br />

![Lanes](README_media/Lanes_screenshot.png "Lanes")
<br />
*Lanes*
<br /><br />

![Lanes with steering wheel](README_media/Lanes_with_steering_wheel.png "Lanes with steering wheel")
<br />
*Lanes with steering wheel*
<br /><br />

#### Click on GIF to open video in YouTube
[![Output when is_demo=True](README_media/Screencast.gif "Output when is_demo=True")](https://www.youtube.com/watch?v=E8GY_svYbQA)
<br />
*Output when is_demo=True*

