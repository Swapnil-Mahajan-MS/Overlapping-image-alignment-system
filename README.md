# SIFT-RANSAC Image Stitcher

A robust image mosaicing system that stitches multiple overlapping images into a seamless composite using SIFT feature detection and RANSAC-based homography estimation.

## Features

- **SIFT Feature Detection**: Automatic keypoint detection and descriptor extraction
- **Robust Matching**: RANSAC algorithm for outlier rejection
- **Homography Estimation**: Accurate geometric transformation computation
- **Bilinear Interpolation**: Smooth image warping and blending
- **Modular Architecture**: Clean, reusable components

## Requirements

```bash
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scikit-image>=0.18.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from skimage.io import imread
from image_mosaicing import create_mosaic

# Load three overlapping images
img1 = imread('img1.png')
img2 = imread('img2.png')  # Reference image (center)
img3 = imread('img3.png')

# Create mosaic
mosaic = create_mosaic(img1, img2, img3)

# Display or save
import matplotlib.pyplot as plt
plt.imshow(mosaic, cmap='gray')
plt.show()
```

## Project Structure

```
sift-ransac-image-stitcher/
├── image_mosaicing.py      # Main module
├── mosaic_usage_example.py # Usage examples
├── sift.py                 # SIFT helper (legacy)
├── requirements.txt
├── README.md
└── images/                 # Input images
    ├── img1.png
    ├── img2.png
    └── img3.png
```

## Algorithm Overview

1. **Feature Extraction**: Detect SIFT keypoints and compute descriptors for all images
2. **Feature Matching**: Match descriptors between image pairs using Brute Force matcher
3. **Homography Estimation**: Use RANSAC to robustly estimate transformation matrices:
   - H₁₂: Maps img2 → img1
   - H₂₃: Maps img2 → img3
4. **Warping & Blending**: Create canvas and blend images using bilinear interpolation

## Advanced Usage

### Custom RANSAC Parameters

```python
from image_mosaicing import ImageMosaic, RANSACEstimator

mosaic_maker = ImageMosaic()
mosaic_maker.ransac = RANSACEstimator(
    epsilon=15.0,      # Error threshold (pixels)
    fraction=0.7,      # Min inlier fraction
    max_iter=5000      # Max iterations
)

result = mosaic_maker.stitch_images(img1, img2, img3)
```

### Visualize Feature Matches

```python
from image_mosaicing import SIFTMatcher
import cv2

matcher = SIFTMatcher()
kp1, desc1 = matcher.detect_and_compute(img1)
kp2, desc2 = matcher.detect_and_compute(img2)
matches = matcher.match_features(desc1, desc2)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
```

## Capturing Your Own Images

For best results when capturing images:

1. **Overlap**: Ensure 30-50% overlap between consecutive images
2. **Distance**: Photograph far-away scenes (reduces parallax issues)
3. **Camera Motion**: Pan horizontally, keep vertical alignment consistent
4. **Resolution**: Resize images to width < 1000 pixels for faster processing
5. **Format**: Convert to grayscale before stitching

Example preprocessing:

```python
img = imread('photo.jpg', as_gray=True)
img_downsampled = img[::5, ::5]  # Downsample by factor of 5
```

## Module Components

### `SIFTMatcher`
Handles feature detection and matching using SIFT algorithm.

### `HomographyEstimator`
Computes 3×3 homography matrices from point correspondences using SVD.

### `RANSACEstimator`
Robustly estimates homography by iteratively finding consensus sets.

### `ImageInterpolator`
Performs bilinear interpolation for smooth image warping.

### `ImageMosaic`
Main orchestrator class that coordinates the entire stitching pipeline.

## Mathematical Background

**Homography Matrix**: Maps points between images via projective transformation:
```
[x']   [h₁ h₂ h₃]   [x]
[y'] ∼ [h₄ h₅ h₆] × [y]
[1 ]   [h₇ h₈ h₉]   [1]
```

**RANSAC Algorithm**:
1. Randomly sample 4 point correspondences
2. Compute homography from sample
3. Count inliers (error < ε)
4. Repeat until sufficient inliers found

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 10.0 | RANSAC error threshold (pixels) |
| `fraction` | 0.8 | Minimum inlier fraction for consensus |
| `max_iter` | 10000 | Maximum RANSAC iterations |

## Limitations

- Requires significant overlap between images
- Works best with far-away scenes (planar assumption)
- Limited to 3 images in current implementation
- No automatic exposure/color correction

## Future Enhancements

- [ ] Support for N images
- [ ] Multi-band blending for seamless transitions
- [ ] Automatic exposure compensation
- [ ] GPU acceleration
- [ ] Real-time video stitching

## Academic Context

Developed for **EE5175: Image Signal Processing, Lab 3**

**Topics covered:**
- Feature detection (SIFT)
- Geometric transformations (Homography)
- Robust estimation (RANSAC)
- Image warping and interpolation

## License

MIT License - Feel free to use for academic and personal projects.

## References

1. Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
2. Fischler, M.A. & Bolles, R.C. (1981). "Random Sample Consensus"
3. Szeliski, R. (2010). "Computer Vision: Algorithms and Applications"

## Troubleshooting

**Issue**: Poor stitching quality
- **Solution**: Increase RANSAC iterations or adjust epsilon

**Issue**: Images don't align
- **Solution**: Ensure proper overlap and image order (left → center → right)

**Issue**: Memory error
- **Solution**: Downsample input images before processing

## Contact

For questions or issues, please open an issue on the repository.
