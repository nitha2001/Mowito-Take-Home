import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Initialize SAM
# Downloaded SAM model check point vit_h from the official website
sam_checkpoint = r"C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def segment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None)
    return image, masks

def calculate_orientation(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return 0
    
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    
    mu11 = moments["mu11"] / moments["m00"]
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    return np.degrees(theta) % 180

def find_best_match(template_masks, test_masks):
    best_iou = 0
    best_mask = None
    
    for test_mask in test_masks:
        for template_mask in template_masks:
            intersection = np.logical_and(test_mask, template_mask).sum()
            union = np.logical_or(test_mask, template_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_mask = test_mask
    
    return best_mask if best_iou > 0.5 else None

def get_dataloader(template_folder, test_folder):
    def extract_paths(folder):
        image_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    template_paths = extract_paths(template_folder)
    test_paths = extract_paths(test_folder)

    return template_paths, test_paths

def load_images(template_paths, test_paths):
    template_images = [cv2.imread(temp_path) for temp_path in template_paths]
    test_images = [cv2.imread(test_path) for test_path in test_paths]
    return template_images, test_images


def process_images(template_path, test_path, output_dir):
    template_img, template_masks = segment_image(template_path)
    template_orientation = calculate_orientation(template_masks[0])
    
    test_img, test_masks = segment_image(test_path)
    
    best_mask = find_best_match(template_masks, test_masks)
    if best_mask is not None:
        test_orientation = calculate_orientation(best_mask)
        rotation = (test_orientation - template_orientation) % 180
        
        contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plt.figure(figsize=(10, 5))
        
        # Template image
        plt.subplot(1, 2, 1)
        plt.imshow(template_img)
        plt.title(f"Template: {os.path.basename(template_path)}")
        plt.axis("off")
        
        # Test image
        plt.subplot(1, 2, 2)
        plt.imshow(test_img)
        plt.title(f"Test: {os.path.basename(test_path)}")
        plt.axis("off")
        
        # Draw contours using Matplotlib
        for contour in contours:
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'g', linewidth=2)
        
        # Add rotation text using Matplotlib
        plt.text(10, 30, f"Rotation: {rotation:.2f} deg", color='white', fontsize=12, 
                 bbox=dict(facecolor='green', alpha=0.7, edgecolor='none', pad=5))
        
        output_filename = f"{os.path.splitext(os.path.basename(template_path))[0]}_{os.path.splitext(os.path.basename(test_path))[0]}_result.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return True
    
    return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\template images-20240530T185119Z-001'
    test_dir = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\test images-20240530T185118Z-001'
    output_dir = r'C:\Users\Nithasree\Downloads\Mowito\Mowito\Take Home\output_images'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files in each directory
    template_images, test_images = get_dataloader(template_dir, test_dir)
    
    total_tests = 0
    successful_tests = 0
    
    for template_path in template_images:
        for test_path in test_images:
            total_tests += 1
            print(f"Processing: Template - {os.path.basename(template_path)}, Test - {os.path.basename(test_path)}")
            
            if process_images(template_path, test_path, output_dir):
                successful_tests += 1
            else:
                print("  No matching object found.")
    
    print(f"\nProcessing complete. Successfully processed {successful_tests} out of {total_tests} image pairs.")

if __name__ == "__main__":
    main()