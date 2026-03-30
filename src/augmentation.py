import cv2
import numpy as np
import random
import os
import glob

def apply_gaussian_noise(image):
    #Additive Gaussian noise with sigma in[5, 20]
    sigma = random.uniform(5.0, 20.0)
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_jpeg_compression(image):
    #Re-encode at quality level randomly selected from 20-80
    quality = random.randint(20, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    return decimg

def apply_dpi_downsampling(image):
    #Reduce resolution from 300 DPI to 150 or 72 DPI, then upscale to maintain dimensions
    target_dpi = random.choice([150, 72])
    scale_factor = target_dpi / 300.0
    h, w = image.shape[:2]
    
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #Upscale back to original size so classifiers don't break on shape mismatch
    restored = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored

def apply_random_cropping(image):
    #Remove 1-3% from each border
    h, w = image.shape[:2]
    #Calculate a different 1-3% crop for each individual side
    top = int(h * random.uniform(0.01, 0.03))
    bottom = int(h * random.uniform(0.01, 0.03))
    left = int(w * random.uniform(0.01, 0.03))
    right = int(w * random.uniform(0.01, 0.03))
    
    cropped = image[top:h-bottom, left:w-right]
    #Resize back to original size
    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored

def apply_bit_depth_reduction(image):
    #Bitwise AND with 11110000 (0xF0) to keep only the top 4 bits
    return image & 0xF0

def generate_augmented_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_dirs =[
        os.path.join(project_root, "google_docs_pdfs_png"),
        os.path.join(project_root, "python_pdfs_png"),
        os.path.join(project_root, "word_pdfs_png")
    ]
    
    output_dir = os.path.join(project_root, "data", "augmented_images")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Loop through each of the 3 class folders
    for in_dir in input_dirs:
        if not os.path.exists(in_dir):
            print(f"Skipping {in_dir} - Folder not found.")
            continue
            
        image_paths = glob.glob(os.path.join(in_dir, "*.png"))
        
        #Get the class name (e.g., "word_pdfs") to prefix the files
        class_name = os.path.basename(in_dir).replace("_png", "")
        
        for path in image_paths:
            filename = os.path.basename(path)
            base_name, ext = os.path.splitext(filename)
            
            #Prefix the filename with the class name so we know what it is for classification!
            new_base_name = f"{class_name}_{base_name}"
            
            #Read as grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            #Save original (1)
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_original{ext}"), img)
            
            #Apply and save augmentations independently (5)
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_gaussian{ext}"), apply_gaussian_noise(img))
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_jpeg{ext}"), apply_jpeg_compression(img))
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_dpidown{ext}"), apply_dpi_downsampling(img))
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_crop{ext}"), apply_random_cropping(img))
            cv2.imwrite(os.path.join(output_dir, f"{new_base_name}_bitdepth{ext}"), apply_bit_depth_reduction(img))
            
    print(f"\nAll augmented images saved to: {output_dir}")

if __name__ == "__main__":
    generate_augmented_dataset()