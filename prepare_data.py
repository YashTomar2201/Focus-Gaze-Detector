import os
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio # The library to read .mat files

# --- CONFIGURATION ---
RAW_DATA_DIR = "Data/MPIIGaze/Data/Normalized" # Make sure this points to where p00, p01 are
OUTPUT_DIR = "CleanDataset"
IMG_SIZE = 64 

def convert_to_degrees(gaze_vector):
    # The dataset gives a 3D vector (x, y, z).
    # We need to turn this into Pitch (up/down) and Yaw (left/right).
    x, y, z = gaze_vector
    
    # Formula to convert 3D Vector -> 2D Angles
    # Note: These formulas depend on the coordinate system, but this is standard for MPII
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    
    return pitch, yaw

def process_mat_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    data_records = []
    
    # 1. Loop through all person folders (p00, p01...)
    for person_id in sorted(os.listdir(RAW_DATA_DIR)):
        person_path = os.path.join(RAW_DATA_DIR, person_id)
        if not os.path.isdir(person_path): continue
        
        print(f"Processing {person_id}...")
        
        # 2. Loop through all .mat files in that folder (day01.mat, day02.mat...)
        for file_name in os.listdir(person_path):
            if not file_name.endswith(".mat"): continue
            
            mat_path = os.path.join(person_path, file_name)
            
            # 3. Load the .mat file
            try:
                mat_contents = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
                
                # The data is usually stored in a variable called 'data'
                if 'data' not in mat_contents:
                    print(f"Skipping {file_name} (No 'data' key found)")
                    continue
                    
                data = mat_contents['data']
                
                # The 'data' object contains 'left' and 'right' eyes. Let's get both.
                # We iterate through the images inside this file
                # 'data.right.image' is an array of eye images
                # 'data.right.gaze' is an array of gaze vectors
                
                sides = ['left', 'right']
                
                for side in sides:
                    if not hasattr(data, side): continue
                    
                    eye_data = getattr(data, side)
                    
                    # Check if we have images and gaze
                    if not hasattr(eye_data, 'image') or not hasattr(eye_data, 'gaze'):
                        continue
                        
                    images = eye_data.image # List of images
                    gazes = eye_data.gaze   # List of 3D vectors
                    
                    # Loop through every image in this .mat file
                    for i in range(len(images)):
                        # 4. Process Image
                        img = images[i] # This is already a numpy array (grayscale)
                        
                        # Normalize and Resize
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        
                        # Save Image
                        save_name = f"{person_id}_{file_name.split('.')[0]}_{side}_{i}.jpg"
                        save_path = os.path.join(OUTPUT_DIR, save_name)
                        cv2.imwrite(save_path, img)
                        
                        # 5. Process Label (3D -> 2D)
                        pitch, yaw = convert_to_degrees(gazes[i])
                        
                        # Add to list
                        data_records.append([save_name, pitch, yaw])
                        
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # 6. Save Labels to CSV
    df = pd.DataFrame(data_records, columns=['filename', 'pitch', 'yaw'])
    df.to_csv("dataset_labels.csv", index=False)
    print(f"\nSUCCESS! Processed {len(df)} eye images.")
    print("You can now run train.py!")

if __name__ == "__main__":
    process_mat_files()