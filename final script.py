import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        print(f"Loading image: {file_path}")
        img = cv2.imread(file_path, 0)
        if img is not None:
            images.append((filename, img))
        else:
            print(f"Warning: Unable to load image {file_path}")
    return images

def template_matching(template, test):
    result = cv2.matchTemplate(test, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    test_color = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    return test_color, top_left, bottom_right

def calculate_rotation_angle(template, test):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(test, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            angle = -np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        else:
            angle = 0.0
    else:
        angle = 0.0

    return angle

def annotate_image(test, top_left, bottom_right, angle):
    roi = test[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Improved thresholding
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the detected region of the test image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        contour[:, :, 0] += top_left[0]
        contour[:, :, 1] += top_left[1]
        cv2.drawContours(test, [contour], -1, (0, 255, 0), 2)
    
    cv2.putText(test, f'Rotation: {angle:.2f} degrees', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return test, angle

def process_images(template_folder, test_folder, output_folder):
    templates = load_images_from_folder(template_folder)
    tests = load_images_from_folder(test_folder)
    
    for t_idx, (t_name, template) in enumerate(templates):
        template_output_folder = os.path.join(output_folder, f'template_{t_idx+1}')
        if not os.path.exists(template_output_folder):
            os.makedirs(template_output_folder)
        
        for test_idx, (test_name, test) in enumerate(tests):
            test_color, top_left, bottom_right = template_matching(template, test)
            angle = calculate_rotation_angle(template, test)
            annotated_image, angle = annotate_image(test_color, top_left, bottom_right, angle)
            
            # Save the annotated image using Matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Rotation: {angle:.2f} degrees')
            plt.axis('off')
            result_path = os.path.join(template_output_folder, f"test{test_idx+1}_template{t_idx+1}.png")
            plt.savefig(result_path)
            plt.close()
            print(f"Processed {t_name} on {test_name}, saved to {result_path}")

if __name__ == "__main__":
    template_folder = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\template images-20240530T185119Z-001\template images'  # Replace with the actual path
    test_folder = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\test images-20240530T185118Z-001\test images'          # Replace with the actual path
    output_folder = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\output_results_final'
    process_images(template_folder, test_folder, output_folder)
