Laboratory Exercise Description — Camera Calibration and Pose Estimation Using ArUco Markers

The objective of this laboratory exercise is to understand and implement camera calibration and real-time pose estimation using ArUco markers. The main things to learn are how to extract intrinsic camera parameters, remove lens distortion, and determine the position and orientation of a known marker board in 3D space.
This exercise demonstrates a practical pipeline for camera calibration and 6-DoF (degrees of freedom) pose estimation using fiducial markers.

The main steps:
1. ArUco marker setup
   - A predefined ArUco dictionary (`DICT_6X6_1000`) is used to generate a GridBoard with known marker dimensions and separations.
   - Detector parameters are initialized for marker detection.
2. Camera calibration
   - Multiple images of the ArUco board are loaded from a directory.
   - Each image is converted to grayscale, and markers are detected with `cv.aruco.ArucoDetector`.
   - Detected marker corners and IDs across all images are collected.
   - OpenCV’s `calibrateCameraAruco` function computes the camera intrinsic matrix and distortion coefficients.
   - The reprojection error is reported to assess calibration accuracy.
3. Pose estimation on video
   - A video containing the ArUco board is processed frame by frame.
   - Marker corners are detected in each frame, and 3D–2D correspondences are matched with the known board geometry.
   - `cv.solvePnP` is used to estimate the rotation and translation vectors of the board relative to the camera.
   - The 3D axes are drawn on the frame to visualize the pose.
   - Lens distortion is corrected using precomputed undistortion maps for display.
4. Visualization
   - Frames are resized and displayed in real-time.
   - Users can quit the visualization by pressing `'q'`.

Additional notes:
* Accuracy depends on the number and quality of calibration images.
* Marker size and separation must match the real-world board.
* Proper lighting and minimal motion blur improve detection and pose estimation.



