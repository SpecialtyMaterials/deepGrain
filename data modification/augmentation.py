import os
import cv2
import numpy as np
import albumentations
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate,
    Blur, OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, RandomBrightnessContrast,
    OneOf, Compose, Rotate, HueSaturationValue
)

augmentations2 = [
        
        RandomBrightnessContrast(brightness_limit=(-0.07,.1), p=1),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-200,100), val_shift_limit=0, p=1.0)
    ]

augmentation2_pipeline = albumentations.Compose(augmentations2)

angleRange = 20

def augmentations1():
    return [
        Rotate(limit=(0, 0), p=1), #unchanged
        Rotate(limit=(45-angleRange, 45+angleRange), p=1), #unchanged
        Rotate(limit=(90, 90), p=1),
        Rotate(limit=(135-angleRange, 135+angleRange), p=1),
        Rotate(limit=(180, 180), p=1),
        Rotate(limit=(225-angleRange, 225+angleRange), p=1),
        Rotate(limit=(270, 270), p=1),
        Rotate(limit=(315-angleRange, 315+angleRange), p=1),
        GridDistortion(num_steps=3, distort_limit=(-.7,0), p=1),
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0)
    ]
    

def load_images(image_dir, edge_dir):
    images = []
    edges = []
    image_files = sorted(os.listdir(image_dir))
    edge_files = sorted(os.listdir(edge_dir))

    for img_file, edge_file in zip(image_files, edge_files):
        img_path = os.path.join(image_dir, img_file)
        edge_path = os.path.join(edge_dir, edge_file)
        image = cv2.imread(img_path)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)  # Assuming edges are grayscale images

        # Ensure they are numpy arrays
        if image is None or edge is None:
            print(f"Failed to load image or edge: {img_path}, {edge_path}")
            continue

        images.append(image)
        edges.append(edge)

    return images, edges

def augment_images1(images, edges, augmentations):
    augmented_images = []
    augmented_edges = []

    for img, edge in zip(images, edges):
        # Ensure they are numpy arrays
        if not isinstance(img, np.ndarray) or not isinstance(edge, np.ndarray):
            print(f"Image or edge is not a numpy array: {type(img)}, {type(edge)}")
            continue

        for aug in augmentations:
            augmented = aug(image=img, mask=edge)
            augmented_images.append(augmented['image'])
            augmented_edges.append(augmented['mask'])

    return augmented_images, augmented_edges

def augment_images2(images, edges):
    augmented_images2 = []
    augmented_edges2 = []

    for img2, edge2 in zip(images, edges):

        augmented2 = augmentation2_pipeline(image=img2, mask=edge2)
        # Ensure they are numpy arrays
        augmented_image = augmented2['image']
        augmented_mask = augmented2['mask']

        augmented_images2.append(augmented_image)
        augmented_edges2.append(augmented_mask)

    return augmented_images2, augmented_edges2

def save_augmented_images(augmented_images, augmented_edges, output_image_dir, output_edge_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_edge_dir, exist_ok=True)

    for i, (img, edge) in enumerate(zip(augmented_images, augmented_edges)):
        img_path = os.path.join(output_image_dir, f"{i+1}.png")
        edge_path = os.path.join(output_edge_dir, f"{i+1}.png")
        cv2.imwrite(img_path, img)
        cv2.imwrite(edge_path, edge)

if __name__ == "__main__":
    image_dir = "C:/Users/nickb/Desktop/projects/cobDetection/images/spliced/splicedImages"
    edge_dir = "C:/Users/nickb/Desktop/projects/cobDetection/images/spliced/splicedEdges"
    output_image_dir = "Training Data/image/raw"
    output_edge_dir = "Training Data/edge/raw"

    images, edges = load_images(image_dir, edge_dir)
    augmentations1 = augmentations1()
    augmented_images, augmented_edges = augment_images1(images, edges, augmentations1)

    augmented_images2, augmented_edges2 = augment_images2(augmented_images, augmented_edges)

    save_augmented_images(augmented_images2, augmented_edges2, output_image_dir, output_edge_dir)