# CameraCalib

CameraCalib is a Python script for calibrating a camera using OpenCV.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install OpenCV.

```bash
pip install opencv-python
```

## Usage

Print a [chessboard pattern](https://github.com/opencv/opencv/blob/master/doc/pattern.png) and make a video recording it slowly, to reduce the motion blur, and from every possible position and angle.

Then save it inside the videos folder and change the variables below as needed.

```python
frames_from_video = True # If true => Extract frames from a video file
undistort = True         # If true => Undistort all calibration images
camera_name = 'note10'   # Name directories, video and output file
extension = 'mp4'        # Video Extension
video = ''               # If empty => videos/{camera_name}.mp4
skip_frames = 25         # Number of frames to skip in video
board_size = (6, 5)      # Chess Board Ratio
square_size = 0.03       # Board's square size in meters
```

For more information see the [OpenCV Documentation](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)