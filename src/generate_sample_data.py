import numpy as np
import SimpleITK as sitk
import os

def generate_sphere(size=64, radius=20):
    """
    Generate a 3D numpy array of a sphere.
    """
    center = size // 2
    x, y, z = np.ogrid[:size, :size, :size]
    
    sphere = (x - center)**2 + (y - center)**2 + (z - center)**2 < radius**2
    return sphere.astype(np.uint8)

def save_nifti(array, path):
    """
    Save a numpy array as a NIfTI file.
    """
    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(image, path)

if __name__ == "__main__":
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    sphere = generate_sphere()
    save_nifti(sphere, os.path.join(data_dir, "sphere.nii.gz"))
    print("Generated and saved sphere.nii.gz in the 'data' directory.")
