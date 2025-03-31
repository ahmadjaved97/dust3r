import sys
sys.path.append("../dust3r")
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from rich import print

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


def enhance_thermal_image(image):
    """
    Enhances a grayscale thermal image using CLAHE, sharpening, and unsharp masking.
    """
    # Step 1: Normalize if not uint8
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 2: Convert to 3-channel RGB
    # image_rgb = np.stack([image] * 3, axis=-1)

    # Step 3: Apply CLAHE on L channel in LAB color space
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(image_lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    image_clahe = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(image_clahe, cv2.COLOR_LAB2RGB)

    # Step 4: Sharpening filter
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Step 5: Unsharp masking
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    final = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return final


def enhance_edges(gray_path, rgb_path):
    # Load images
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(rgb_path)

    # Resize gray if needed to match RGB
    if gray.shape != rgb.shape[:2]:
        gray = cv2.resize(gray, (rgb.shape[1], rgb.shape[0]))

    #Apply CLAHE to the gray image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    #Convert RGB to grayscale and extract edges
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_rgb, threshold1=50, threshold2=150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    #  Overlay edges (white) onto CLAHE image
    enhanced = gray_clahe.copy()
    enhanced[edges_dilated > 0] = 255

    return enhanced

def enhance_edges(gray):

    #Apply CLAHE to the gray image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    return gray_clahe

class FreiburgDatasetThermal(BaseStereoViewDataset):
    """
    PyTorch Dataset class for Freiburg Thermal Dataset.
    Dynamically loads IR (or optionally RGB), depth maps, intrinsics, and extrinsics.
    """
    def __init__(self, root_dir, dataset_filename, split="train", use_rgb=False,use_enhance=True, clahe=False, resolution=224, *args, **kwargs):
        assert resolution is not None, "Error: resolution must be defined!"

        self.root_dir = root_dir
        self.use_rgb = use_rgb
        self.use_enhance = use_enhance
        self.clahe = clahe
        super().__init__(split=split, resolution=resolution, *args, **kwargs)

        self.dataset_label = "FreiburgThermal"

        json_path = os.path.join(root_dir, dataset_filename)
        with open(json_path, "r") as f:
            self.metadata = json.load(f)

        num_samples = len(self.metadata)
        print('Total samples: ', num_samples)

        # ---- Example pairing strategy: (i, i+1) ----
        # 2 pair
        # self.combinations = [(i, i+1) for i in range(0, len(self.metadata), 2) if i+1 < len(self.metadata)]
        # 3 pair
        self.combinations = [(i, j) for i in range(len(self.metadata)) for j in (i+1, i+2, i+3)  if j < len(self.metadata)]
        # 4 pair
        # self.combinations = [(i, j) for i in range(len(self.metadata)) for j in range(i + 1, i + 5) if j < len(self.metadata)]
        # 6 pair
        # self.combinations = [(i, j) for i in range(len(self.metadata)) for j in range(i, i + 6) if j < len(self.metadata)]


        print(f"[FreiburgDataset] Using {len(self.metadata)} entries and {len(self.combinations)} pairs for split='{split}'")

    def __len__(self):
        return len(self.combinations)

    def _get_metadatapath(self, idx):
        return self.metadata[idx]

    def _get_impath(self, idx):
        sample = self.metadata[idx]
        return sample["rgb_path"] if self.use_rgb else sample["ir_path"]

    def _get_depthpath(self, idx):
        return self.metadata[idx]["depth_map_path"]

    def _read_depthmap(self, depthpath):
        depth_data = np.load(depthpath).reshape(224, 224)
        return depth_data.astype(np.float32)

    def _get_views(self, idx, resolution, rng):
        # img1_idx, img2_idx = self.combinations[idx % len(self.combinations)]
        img1_idx, img2_idx = self.combinations[idx]

        views = []
        for im_idx in [img1_idx, img2_idx]:
            metadata = self._get_metadatapath(im_idx)
            impath = self._get_impath(im_idx)
            depthpath = self._get_depthpath(im_idx)

            K = np.array(metadata["intrinsics"], dtype=np.float32)
            camera_pose_4x4 = np.array(metadata["extrinsics"], dtype=np.float32)

            image = imread_cv2(impath, cv2.IMREAD_GRAYSCALE if not self.use_rgb else cv2.IMREAD_UNCHANGED)

            if not self.use_rgb and self.clahe:
                image = enhance_edges(image)

            if not self.use_rgb:
                image = np.stack([image]*3, axis=-1)  # Grayscale -> 3-channel manually
                
            depthmap = self._read_depthmap(depthpath)
            original_shape = tuple(metadata["shape"])

            image = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            if not self.use_rgb and self.use_enhance:
                image = enhance_thermal_image(image)


            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            

            image, depthmap, K = self._crop_resize_if_necessary(
                image, depthmap, K, resolution, rng=rng, info=impath
            )


            views.append({
                "img": image,
                "depthmap": depthmap,
                "camera_intrinsics": K,
                "camera_pose": camera_pose_4x4,
                "dataset": self.dataset_label,
                "instance": os.path.basename(impath),
            })

        return views


if __name__ == "__main__":
    dataset_ir = FreiburgDatasetThermal(root_dir="../data/cuda_lab", dataset_filename="dataset_v1_224_train.json", split="train", use_rgb=False,use_enhance=False,clahe=False, resolution=224)

    sample_ir = dataset_ir[0]
    print('Total dataset length: ', len(dataset_ir))
    # print('Idxs for last index of dataset: ', dataset_ir[1])
    
    for i in range(90, 100):
        sample_ir = dataset_ir[i]
        img = sample_ir[0]["img"].squeeze(0).detach().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(f"sample{i}.jpg", img)

        depthmap = sample_ir[0]["depthmap"]
        # Normalize to 0–255 for visualization
        depth_norm = cv2.normalize(depthmap, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(f"depthmap{i}.png", depth_colored)


    # print("IR Image Shape:", sample_ir[0]["img"].shape)
    # print("Depth Shape:", sample_ir[0]["depthmap"].shape)
