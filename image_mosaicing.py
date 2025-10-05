"""
Image Mosaicing Module
Implements SIFT-based image stitching with RANSAC homography estimation
"""

import numpy as np
import cv2
from typing import Tuple, List
import math


class SIFTMatcher:
    """Handle SIFT feature detection and matching"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple:
        """Extract SIFT keypoints and descriptors"""
        return self.sift.detectAndCompute(image, None)
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match descriptors between two images"""
        matches = self.bf.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)
    
    def extract_point_pairs(self, keypoints1, keypoints2, matches) -> List:
        """Extract matched point pairs from keypoints"""
        point_pairs = []
        for match in matches:
            pt1 = keypoints1[match.queryIdx].pt
            pt2 = keypoints2[match.trainIdx].pt
            point_pairs.append(([pt1], [pt2]))
        return point_pairs


class HomographyEstimator:
    """Compute homography matrices"""
    
    @staticmethod
    def compute_homography(points: List, corresponding_points: List) -> np.ndarray:
        """
        Compute homography matrix from point correspondences
        
        Args:
            points: List of points in source image
            corresponding_points: List of corresponding points in target image
            
        Returns:
            3x3 homography matrix
        """
        n = len(points)
        A = np.zeros((2*n, 9))
        
        for i in range(n):
            x, y = points[i][0][0], points[i][0][1]
            xp, yp = corresponding_points[i][0][0], corresponding_points[i][0][1]
            
            A[2*i] = [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]
            A[2*i+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp]
        
        # Solve using SVD (constrained least squares)
        _, _, v = np.linalg.svd(A.T @ A)
        H = v[-1].reshape(3, 3)
        return H
    
    @staticmethod
    def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Apply homography to a point and normalize"""
        result = H @ point
        if result[2] != 0:
            result = result / result[2]
        return result


class RANSACEstimator:
    """RANSAC-based robust homography estimation"""
    
    def __init__(self, epsilon: float = 10.0, fraction: float = 0.8, max_iter: int = 10000):
        self.epsilon = epsilon
        self.fraction = fraction
        self.max_iter = max_iter
        self.homography_estimator = HomographyEstimator()
    
    def estimate(self, point_pairs: List) -> np.ndarray:
        """
        Estimate homography using RANSAC
        
        Args:
            point_pairs: List of matched point pairs
            
        Returns:
            Best homography matrix found
        """
        best_H = None
        max_inliers = 0
        iterations = 0
        
        while iterations < self.max_iter:
            # Randomly sample 4 point pairs
            import random
            sample = random.sample(point_pairs, 4)
            
            source_pts = [pt[1] for pt in sample]
            target_pts = [pt[0] for pt in sample]
            
            # Compute homography from sample
            H = self.homography_estimator.compute_homography(source_pts, target_pts)
            
            # Count inliers
            inliers = self._count_inliers(H, point_pairs)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
            
            # Early stopping if enough inliers found
            if max_inliers > self.fraction * len(point_pairs):
                break
                
            iterations += 1
        
        return best_H if best_H is not None else np.eye(3)
    
    def _count_inliers(self, H: np.ndarray, point_pairs: List) -> int:
        """Count inliers for given homography"""
        count = 0
        for pt_pair in point_pairs:
            vec = [pt_pair[1][0][0], pt_pair[1][0][1], 1]
            transformed = H @ vec
            transformed = transformed / transformed[2] if transformed[2] != 0 else transformed
            
            actual = [pt_pair[0][0][0], pt_pair[0][0][1], 1]
            error = np.linalg.norm(transformed - actual)
            
            if error < self.epsilon:
                count += 1
        
        return count


class ImageInterpolator:
    """Bilinear interpolation for image warping"""
    
    @staticmethod
    def interpolate(image: np.ndarray, x: float, y: float) -> Tuple[float, bool]:
        """
        Perform bilinear interpolation
        
        Args:
            image: Source image
            x, y: Coordinates to interpolate
            
        Returns:
            Interpolated value and validity flag
        """
        h, w = image.shape[:2]
        x_floor = math.floor(x)
        y_floor = math.floor(y)
        
        # Check bounds
        if not (0 <= x_floor < h-1 and 0 <= y_floor < w-1):
            return 0.0, False
        
        a = x - x_floor
        b = y - y_floor
        
        # Bilinear interpolation formula
        value = ((1-a)*(1-b)*image[x_floor, y_floor] +
                 (1-a)*b*image[x_floor, y_floor+1] +
                 a*(1-b)*image[x_floor+1, y_floor] +
                 a*b*image[x_floor+1, y_floor+1])
        
        return value, True


class ImageMosaic:
    """Main class for creating image mosaics"""
    
    def __init__(self):
        self.matcher = SIFTMatcher()
        self.ransac = RANSACEstimator()
        self.interpolator = ImageInterpolator()
    
    def stitch_images(self, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray) -> np.ndarray:
        """
        Stitch three images into a mosaic (img2 is reference)
        
        Args:
            img1, img2, img3: Input images (grayscale)
            
        Returns:
            Stitched mosaic image
        """
        # Convert to uint8 if needed
        img1 = self._ensure_uint8(img1)
        img2 = self._ensure_uint8(img2)
        img3 = self._ensure_uint8(img3)
        
        # Extract features and match
        kp1, desc1 = self.matcher.detect_and_compute(img1)
        kp2, desc2 = self.matcher.detect_and_compute(img2)
        kp3, desc3 = self.matcher.detect_and_compute(img3)
        
        matches12 = self.matcher.match_features(desc1, desc2)
        matches23 = self.matcher.match_features(desc3, desc2)
        
        # Extract point pairs
        points12 = self.matcher.extract_point_pairs(kp1, kp2, matches12)
        points23 = self.matcher.extract_point_pairs(kp3, kp2, matches23)
        
        # Estimate homographies using RANSAC
        H12 = self.ransac.estimate(points12)  # img2 -> img1
        H23 = self.ransac.estimate(points23)  # img2 -> img3
        
        # Create mosaic
        mosaic = self._create_canvas(img1, img2, img3, H12, H23)
        
        return mosaic
    
    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 format"""
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img
    
    def _create_canvas(self, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray,
                      H12: np.ndarray, H23: np.ndarray) -> np.ndarray:
        """Create the final mosaic canvas"""
        # Determine canvas size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        
        canvas_h = h1 + h2 + h3
        canvas_w = w1 + w2 + w3
        canvas = np.zeros((canvas_h, canvas_w))
        
        # Compute offsets
        offset_h = int(np.ceil(canvas_h / 3))
        offset_w = int(np.ceil(canvas_w / 3))
        
        # Fill canvas
        for j in range(canvas_w):
            for i in range(canvas_h):
                y = i - offset_h
                x = j - offset_w
                vec = np.array([x, y, 1])
                
                # Map to each image
                pt1 = H12 @ vec
                pt1 = pt1 / pt1[2] if pt1[2] != 0 else pt1
                
                pt3 = H23 @ vec
                pt3 = pt3 / pt3[2] if pt3[2] != 0 else pt3
                
                # Interpolate values
                val1, valid1 = self.interpolator.interpolate(img1, pt1[1], pt1[0])
                val2, valid2 = self.interpolator.interpolate(img2, y, x)
                val3, valid3 = self.interpolator.interpolate(img3, pt3[1], pt3[0])
                
                # Blend values
                total_valid = valid1 + valid2 + valid3
                if total_valid > 0:
                    blended = (valid1*val1 + valid2*val2 + valid3*val3) / total_valid
                    canvas[i, j] = blended
        
        return canvas


def create_mosaic(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray) -> np.ndarray:
    """
    Convenience function to create image mosaic
    
    Args:
        img1, img2, img3: Three overlapping images (img2 is reference)
        
    Returns:
        Stitched mosaic image
    """
    mosaic = ImageMosaic()
    return mosaic.stitch_images(img1, img2, img3)
