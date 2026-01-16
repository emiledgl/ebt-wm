import os
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms import v2

class DROIDDataset(torch.utils.data.Dataset):
    """DROID dataset using LeRobotDataset backend."""

    def __init__(
        self,
        root="data",
        num_episodes=300,
        tubelets_per_clip=8,
        image_size=256,
        tubelet_size=2,
        camera_keys=None,
        normalize=False,
        video_backend="torchcodec",
    ):
        self.tubelets_per_clip = tubelets_per_clip
        self.image_size = image_size
        self.tubelet_size = tubelet_size
        self.normalize = normalize
        self.total_frames = tubelets_per_clip * tubelet_size
        
        # Default camera keys
        if camera_keys is None:
            camera_keys = [
                "observation.images.wrist_left",
                "observation.images.exterior_1_left",
                "observation.images.exterior_2_left",
            ]
        self.camera_keys = camera_keys

        fps = 15 # DROID default
        delta_timestamps = {
            key: [i / fps for i in range(self.total_frames)]
            for key in self.camera_keys + ["observation.state", "action"]
        }

        self.dataset = LeRobotDataset(
            "lerobot/droid_1.0.1",
            root=os.path.join(root, "droid_1.0.1"),
            episodes=list(range(num_episodes)),
            image_transforms=v2.Resize((image_size, image_size)),
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )

    def _normalize(self, tensor: torch.Tensor, key: str):
        if "images" in key:
            normalize_transform = v2.Normalize(
                mean=self.dataset.meta.stats[key]['mean'].squeeze().tolist(),
                std=self.dataset.meta.stats[key]['std'].squeeze().tolist(),
            )
            return normalize_transform(tensor)
        else:
            mean = torch.tensor(self.dataset.meta.stats[key]['mean'], device=tensor.device, dtype=tensor.dtype).view(1, -1)
            std = torch.tensor(self.dataset.meta.stats[key]['std'], device=tensor.device, dtype=tensor.dtype).view(1, -1)
            return (tensor - mean) / std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        
        # Randomly select a camera
        video = []
        for camera_key in self.camera_keys:
            video_view = data[camera_key]
            video.append(video_view)
        video = torch.stack(video, dim=1) # [N*tubelet_size, V, C, H, W]
        proprios = data["observation.state"] # [N*tubelet_size, P]

        # Extract Actions [N-1, A]
        action_indices = torch.arange(self.tubelet_size - 1, self.total_frames, self.tubelet_size)[:-1]
        actions = data["action"][action_indices]

        if self.normalize:
            video = self._normalize(video, self.camera_keys[0])
            proprios = self._normalize(proprios, "observation.state")
            actions = self._normalize(actions, "action")

        return video, proprios, actions
    

if __name__ == "__main__":
    dataset = DROIDDataset(
        tubelets_per_clip=8,
        image_size=256,
        tubelet_size=2,
        num_episodes=100,
        camera_keys=None,
        normalize=True,
        video_backend="pyav",
    )

    print(f"Dataset Length: {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        video_batch, proprios_batch, actions_batch = batch
        print(f"Video Batch Shape: {video_batch.shape}")
        print(f"Proprios Batch Shape: {proprios_batch.shape}")
        print(f"Actions Batch Shape: {actions_batch.shape}")
        break