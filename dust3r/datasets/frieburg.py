import sys

# sys.path.append("/home/user/javeda1/dust3r")
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


class FreiburgDataset(BaseStereoViewDataset):
    """
    PyTorch Dataset class for Freiburg Thermal Dataset.
    Dynamically loads IR (or optionally RGB), depth maps, intrinsics, and extrinsics.
    """
    def __init__(self, root_dir, split="train", use_rgb=False, resolution=224, *args, **kwargs):
        """
        :param root_dir: Path to dataset directory
        :param split: "train", "val", or "test"
        :param use_rgb: If True, loads RGB images; otherwise, loads IR images (default: False)
        :param resolution: Target image resolution (Required)
        """
        assert resolution is not None, "🚨 Error: resolution must be defined!"

        self.root_dir = root_dir
        self.use_rgb = use_rgb
        super().__init__(split=split, resolution=resolution, *args, **kwargs)  # ✅ Ensure resolution is passed

        self.dataset_label = "FreiburgThermal"

        # Load dataset metadata (JSON)
        json_path = os.path.join(root_dir, "dataset_info.json")
        with open(json_path, "r") as f:
            self.metadata = json.load(f)

        # Filter dataset based on `split`
        num_samples = len(self.metadata)
        print('total samples: ', num_samples)
        train_split = int(0.8 * num_samples)
        val_split = int(0.9 * num_samples)

        
        if split == "train":
            self.metadata = self.metadata[:train_split]
        elif split == "val":
            self.metadata = self.metadata[train_split:val_split]
        else:
            self.metadata = self.metadata[val_split:]
        
        # Frame selection strategy
        self.combinations = [(i, i + 2) for i in range(0, len(self.metadata) - 2, 2)]

        print(f"[FreiburgDataset] Using {len(self.metadata)} entries and {len(self.combinations)} pairs for split='{split}'")

        # ✅ Define transformations dynamically based on IR vs RGB
        if self.use_rgb:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ✅ RGB normalization
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # ✅ Expand grayscale to 3 channels
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ✅ Normalize IR like RGB
            ])

    def make_4x4(self, matrix_3x4):
        """Convert a 3x4 projection matrix to a 4x4 transformation matrix."""
        B = matrix_3x4.shape[0] if matrix_3x4.ndim == 3 else 1
        last_row = torch.tensor([[0, 0, 0, 1]], dtype=matrix_3x4.dtype, device=matrix_3x4.device)
        last_row = last_row.repeat(B, 1, 1) if B > 1 else last_row  # Ensure batch compatibility
        matrix_4x4 = torch.cat([matrix_3x4, last_row], dim=-2)  # Append as the last row
        return matrix_4x4
    
    def __len__(self):
        return len(self.metadata)

    def _get_metadatapath(self, idx):
        """Returns the metadata JSON path for an image."""
        return self.metadata[idx]

    def _get_impath(self, idx):
        """Returns the image path (IR or RGB) based on the setting."""
        sample = self.metadata[idx]
        return sample["rgb_path"] if self.use_rgb else sample["ir_path"]

    def _get_depthpath(self, idx):
        """Returns the depth map path."""
        return os.path.join("/home/s63ajave_hpc/", self.metadata[idx]["depth_map_path"])

    def _read_depthmap(self, depthpath):
        """Loads and returns depth map."""
        depth_data = np.load(depthpath).reshape(224, 224)
        return depth_data.astype(np.float32)

    def _get_views(self, idx, resolution, rng):
        """
        Loads two views (image, depth, intrinsics, extrinsics) using frame selection strategy.
        """
        img1_idx, img2_idx = self.combinations[idx % len(self.combinations)]

        views = []
        for im_idx in [img1_idx, img2_idx]:
            metadata = self._get_metadatapath(im_idx)
            impath = self._get_impath(im_idx)
            depthpath = self._get_depthpath(im_idx)

            # Load intrinsics and extrinsics
            K = np.array(metadata["intrinsics"], dtype=np.float32)  # ✅ Convert to NumPy
            R = np.array(metadata["extrinsics"]["rotation"], dtype=np.float32)
            t = np.array(metadata["extrinsics"]["translation"], dtype=np.float32)



            # ✅ Convert 3x4 Camera Pose to 4x4
            # ✅ Convert 3x4 Camera Pose to 4x4 NumPy Matrix
            camera_pose_4x4 = np.eye(4, dtype=np.float32)  # Identity matrix
            camera_pose_4x4[:3, :3] = R  # ✅ Assign Rotation
            camera_pose_4x4[:3, 3] = t.squeeze()  # ✅ Assign Translation


            # Load images and depth
            image = imread_cv2(impath, cv2.IMREAD_GRAYSCALE if not self.use_rgb else cv2.IMREAD_COLOR)
            if self.use_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not self.use_rgb:
                # Histogram equalization for grayscale (thermal) image
                image = cv2.equalizeHist(image)
            # Convert to float and rescale to [0, 1]
            # image = image/ 255.0

            depthmap = self._read_depthmap(depthpath)

            # ✅ Get `original_shape` from JSON
            original_shape = tuple(metadata["original_shape"])  # Expected (H, W)

            # ✅ Resize Depth Map and IR Image to `original_shape` Before Further Processing
            image = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            # depthmap = cv2.resize(depthmap, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


            # ✅ Convert Image to PIL Format
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)  # Convert NumPy array to PIL image


            # # Resize image & depth map if needed
            # image, depthmap, K = self._crop_resize_if_necessary(
            #     image, depthmap, K, resolution, rng=rng, info=impath
            # )

            views.append({
                "img": image,
                "depthmap": depthmap,
                "camera_intrinsics": K,
                "camera_pose":  camera_pose_4x4,  # Convert R, t to 3x4 Projection matrix
                "dataset": self.dataset_label,
                "instance": os.path.basename(impath),  # ✅ Use filename as instance
            })

        return views


# Example usage
if __name__ == "__main__":
    dataset_ir = FreiburgDataset(root_dir="/home/user/javeda1/ThermalVision3D", split="train", use_rgb=False, resolution=224)
    dataset_rgb = FreiburgDataset(root_dir="/home/user/javeda1/ThermalVision3D", split="train", use_rgb=True, resolution=224)

    sample_ir = dataset_ir[0]
    sample_rgb = dataset_rgb[0]

    # print("IR Sample Keys:", sample_ir)
    print("IR Image Shape:", sample_ir[0]["img"].shape)  # ✅ Correct: Access first view
    print("Depth Shape:", sample_ir[0]["depthmap"].shape)

    # print("RGB Sample Keys:", sample_rgb.keys())
    # print("RGB Image Shape:", sample_rgb["image"].shape)
    # print("Depth Shape:", sample_rgb["depthmap"].shape)

