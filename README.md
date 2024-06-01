# TO RIN THE CODE, DO CHANGE THE PATHS OF TEMPLATE IMAGES, TEST IMAGES AND OUTPUT IMAGES IN THE MAIN FUNCTION

 Techniques and Their Effectiveness
 Objective
The objective is to identify the position and orientation of a given object in test images using template images. The output should display the test image with a polygon marking the object's boundary and text indicating the rotation angle between the object in the template and test image.

Techniques Tried
1. ORB Feature Matching with Contour Detection
2. SIFT Feature Matching with Improved Contour Detection
3. Segment Anything Model (SAM) with Mask Matching

 Detailed Analysis of Each Technique

# 1. ORB Feature Matching with Contour Detection

Description:
- Feature Detection and Matching: ORB (Oriented FAST and Rotated BRIEF) was used to detect and compute features in the images. The features from the template and test images were matched using the BFMatcher with Hamming distance.
- Rotation Calculation: The matches were used to find the homography matrix, which helps in calculating the rotation angle between the template and test images.
- Contour Detection: Canny edge detection was applied to the region of interest (ROI) in the test image, followed by finding and drawing contours.

Problems Faced:
- Suboptimal Performance: The performance of ORB was not satisfactory. ORB did not handle complex scenes well, often resulting in poor feature matching. This led to inaccurate detection of the object's position and orientation. In many cases, the detected features were insufficient for reliable matching, causing significant errors in the computed rotation angles.

Example Output:

  - Rotation: The calculated rotation angle was often incorrect due to poor feature matching.
 
# 2. SIFT Feature Matching with Improved Contour Detection

Description:
- Feature Detection and Matching: SIFT (Scale-Invariant Feature Transform) was used for detecting and computing features in the images. The features from the template and test images were matched using the BFMatcher with L2 norm.
- Rotation Calculation: The matches were used to find the homography matrix, helping in calculating the rotation angle between the template and test images.
- Contour Detection: The ROI was converted to grayscale, blurred using Gaussian blur, and then thresholded using Otsu's method. Contours were found and drawn on the test image.

Problems Faced:
- Not Perfect but Better: SIFT provided better results compared to ORB. However, it was not without issues. In some cases, especially with noisy or low-contrast images, the feature detection and matching were not as reliable. The computation of the rotation angle was more accurate than ORB, but it still had occasional errors.

Example Output:
 

  - Rotation: The rotation angle was more accurate compared to ORB, but occasional mismatches still occurred.

# 3. Segment Anything Model (SAM) with Mask Matching

Description:
- Segmentation: The SAM model was used to segment the template and test images, producing masks for the objects.
- Orientation Calculation: Moments of the masks were used to calculate the center of mass and orientation of the objects.
- Mask Matching: The best matching mask from the test image was found by comparing Intersection over Union (IoU) with the template mask. The rotation angle was calculated based on the orientation difference.

Problems Faced:
- Segmentation Issues: The SAM model did not segment the images properly, leading to incorrect rotation calculations. The model struggled with certain images, failing to produce accurate masks. This was a critical issue as the subsequent steps relied on accurate segmentation.
- Need for Preprocessing: If more time were available, preprocessing the images (e.g., enhancing contrast, removing noise) could have improved the segmentation quality significantly. Without preprocessing, the model's performance was subpar, making it unsuitable for this task in its current form.

Example Output:
   - Rotation: Due to segmentation errors, the rotation angles were often inaccurate or completely incorrect.

 Conclusion

The analysis revealed that each technique had its own set of challenges. ORB struggled with complex scenes and provided poor feature matching. SAM, while promising, failed to segment images accurately without preprocessing, leading to incorrect rotation calculations. SIFT, despite its issues, provided the best balance of accuracy and robustness among the three methods. Therefore, the final implementation uses SIFT for feature matching and improved contour detection to identify the object's position and orientation.

Chosen Technique: SIFT Feature Matching with Improved Contour Detection
 Summary of Problems and Solutions

1. SAM:
   - Problem: Poor segmentation results leading to incorrect rotation calculations.
   - Solution: Preprocessing the images to enhance segmentation quality could improve results.

2. ORB:
   - Problem: Suboptimal performance in complex scenes with poor feature matching.
   - Solution: Using a more robust feature detection algorithm like SIFT.

3. SIFT:
   - Problem: Occasional mismatches in noisy or low-contrast images.
   - Solution: While not perfect, SIFT provided better results than ORB and SAM, making it the best choice among the tested methods. Further improvements could be achieved with more sophisticated preprocessing techniques.
