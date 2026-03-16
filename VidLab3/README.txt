Laboratory Exercise Description — Image Stitching and Panorama Assembly:

Objective is learning how to automatically combine multiple overlapping images into a single panorama using keypoint detection, descriptor matching, homography estimation, and different blending techniques.

Main steps of the process:
1. Keypoint detection and description
   - SIFT is used to detect distinctive points and compute descriptors in each image. Optional scaling can speed up detection.
2. Descriptor matching
   - FLANN matcher combined with Lowe’s ratio test (ratio_thresh ≈ 0.7) filters out false matches.
3. Homography estimation
   - Robust homography is computed using `cv.findHomography(..., RANSAC, ransac_thresh)` to align images while rejecting outliers.
4. Canvas preparation and transformation
   - The corners of each image are transformed relative to the middle image to determine the canvas size and translation offsets (tx, ty).
5. Warping and blending (panorama assembly)
   - Two blending methods are implemented:
     1) average — simple average of overlapping pixels, which can create visible seams
     2) feather — weighted blending where each image has higher influence at its center and gradually decreases toward edges using a distance transform, producing smoother transitions
6. Final result
   - Panorama image is saved and displayed.



