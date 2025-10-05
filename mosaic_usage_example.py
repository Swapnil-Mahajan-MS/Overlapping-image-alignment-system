"""
Example usage of the Image Mosaicing Module
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from image_mosaicing import create_mosaic, ImageMosaic

# Example 1: Simple usage with provided images
def example_basic():
    """Basic mosaicing example"""
    # Load images
    img1 = imread('./img1.png')
    img2 = imread('./img2.png')
    img3 = imread('./img3.png')
    
    # Create mosaic (img2 is reference)
    mosaic = create_mosaic(img1, img2, img3)
    
    # Display result
    plt.figure(figsize=(12, 12))
    plt.imshow(mosaic, cmap='gray')
    plt.title('Image Mosaic')
    plt.axis('off')
    plt.show()
    
    return mosaic


# Example 2: Advanced usage with custom parameters
def example_advanced():
    """Advanced example with custom RANSAC parameters"""
    from image_mosaicing import ImageMosaic, RANSACEstimator
    
    # Load images
    img1 = imread('./img1.png')
    img2 = imread('./img2.png')
    img3 = imread('./img3.png')
    
    # Create mosaic object with custom RANSAC parameters
    mosaic_maker = ImageMosaic()
    mosaic_maker.ransac = RANSACEstimator(
        epsilon=15.0,      # Error threshold
        fraction=0.7,      # Inlier fraction
        max_iter=5000      # Max iterations
    )
    
    # Stitch images
    result = mosaic_maker.stitch_images(img1, img2, img3)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(result, cmap='gray')
    plt.title('Custom RANSAC Parameters')
    plt.axis('off')
    plt.show()
    
    return result


# Example 3: Working with own captured images
def example_custom_images():
    """Process custom captured images"""
    # Load and downsample images
    img1 = imread('./i1.jpg', as_gray=True)
    img2 = imread('./i2.jpg', as_gray=True)
    img3 = imread('./i3.jpg', as_gray=True)
    
    # Downsample by factor of 5
    img1 = img1[::5, ::5]
    img2 = img2[::5, ::5]
    img3 = img3[::5, ::5]
    
    # Create mosaic
    mosaic = create_mosaic(img1, img2, img3)
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2 (Reference)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('Image 3')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(mosaic, cmap='gray')
    plt.title('Mosaic Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return mosaic


# Example 4: Visualizing feature matches
def example_visualize_matches():
    """Visualize SIFT feature matches"""
    import cv2
    from image_mosaicing import SIFTMatcher
    
    # Load images
    img1 = imread('./img1.png')
    img2 = imread('./img2.png')
    
    # Ensure uint8
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype('uint8')
        img2 = (img2 * 255).astype('uint8')
    
    # Extract and match features
    matcher = SIFTMatcher()
    kp1, desc1 = matcher.detect_and_compute(img1)
    kp2, desc2 = matcher.detect_and_compute(img2)
    matches = matcher.match_features(desc1, desc2)
    
    # Draw matches
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, 
        matches[:50], None, flags=2
    )
    
    plt.figure(figsize=(20, 10))
    plt.imshow(matched_img, cmap='gray')
    plt.title('SIFT Feature Matches (Top 50)')
    plt.axis('off')
    plt.show()


# Example 5: Save mosaic to file
def example_save_mosaic():
    """Create and save mosaic"""
    from skimage.io import imsave
    
    # Load images
    img1 = imread('./img1.png')
    img2 = imread('./img2.png')
    img3 = imread('./img3.png')
    
    # Create mosaic
    mosaic = create_mosaic(img1, img2, img3)
    
    # Save result
    imsave('mosaic_result.png', mosaic.astype(np.uint8))
    print("Mosaic saved as 'mosaic_result.png'")
    
    return mosaic


if __name__ == "__main__":
    # Run basic example
    print("Creating image mosaic...")
    mosaic = example_basic()
    print(f"Mosaic shape: {mosaic.shape}")
    print("Done!")
