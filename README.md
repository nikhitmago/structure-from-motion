# structure-from-motion
Reconstructing the 3-D positions of a set of matching points in the images and inferring the camera extrinsic parameters

## Steps for SFM Pipeline:

- Keypoint Feature Extraction using SIFT
- Feature Matching using BruteForceMatcher
- Finding Essential Matrix using RANSAC global matching
- Decomposing Essential Matrix into (R, t) components
- Triangulation

## Visualization

- The 3D points can be visualized in softwares like Meshlab where you can easily upload the `output.obj` file that is generated after running `python3 sfm.py`
- The program outputs the rotation, translation, and projection matrices for each pair of images in the data folder
