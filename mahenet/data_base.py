import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
IGNORE_INDEX=255
SOURCE_TO_TARGET={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 13: 9, 16: 10, 17: 11, 18: 12, 20: 13, 21: 14, 24: 15, 26: 16, 27: 17, 29: 18, 30: 19, 32: 20, 33: 21, 34: 22, 35: 23, 37: 24, 38: 25, 41: 26, 43: 27, 44: 28, 46: 29, 47: 30, 48: 31, 49: 32, 50: 33, 51: 34, 52: 35, 53: 36}
class DMS37Dataset(Dataset):
    """Fixed DMS37 mapping and image lists."""

    def __init__(self, root, split, image_size=512, augment=False, manifest=None, label_cache=None):
        if label_cache:
            raise ValueError("DMS-52 does not use a label cache.")
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_size = image_size
        self.augment = augment
        if not manifest and (not self.image_dir.is_dir() or not self.label_dir.is_dir()):
            raise RuntimeError(f"Official DMS split is unavailable: {split}")
        with (self.root / "taxonomy.json").open() as file:
            taxonomy = json.load(file)
        if len(taxonomy["names"]) != 57:
            raise RuntimeError("Unexpected DMS taxonomy size; expected 57 source entries.")
        self.colors = np.asarray(taxonomy["srgb_colormap"], dtype=np.uint8)
        self.label_lookup = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for source_id, target_id in SOURCE_TO_TARGET.items():
            self.label_lookup[source_id] = target_id
        if manifest:
            with Path(manifest).open() as file:
                records = json.load(file)["splits"][split]
            self.samples = [(self.root / record["image_path"], self.root / record["label_path"]) for record in records]
        else:
            with (self.root / "info.json").open() as file:
                metadata = json.load(file)
            valid_ids = set(SOURCE_TO_TARGET)
            eligible_stems = {
                Path(record["image_path"]).stem
                for record in metadata
                if record["split"] == split and any(record["label_fraction"][source_id] > 0 for source_id in valid_ids)
            }
            self.samples = sorted(
                (image_path, self.label_dir / f"{image_path.stem}.png")
                for image_path in self.image_dir.glob("*.*")
                if image_path.stem in eligible_stems and (self.label_dir / f"{image_path.stem}.png").exists()
            )
        if not self.samples:
            raise RuntimeError(f"No valid DMS-52 image/label pairs in {self.image_dir}")

    def _encode_label(self, image):
        if image.mode == "L":
            source = np.asarray(image, dtype=np.uint8)
        else:
            rgb = np.asarray(image, dtype=np.uint8)
            source = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
            for index, color in enumerate(self.colors):
                source[np.all(rgb == color, axis=-1)] = index
        return Image.fromarray(self.label_lookup[source])

    def _resize_pair(self, image, label):
        scale = self.image_size / max(image.size)
        width, height = round(image.size[0] * scale), round(image.size[1] * scale)
        image = TF.resize(image, [height, width], InterpolationMode.BILINEAR)
        label = TF.resize(label, [height, width], InterpolationMode.NEAREST)
        left, top = (self.image_size - width) // 2, (self.image_size - height) // 2
        padding = [left, top, self.image_size - width - left, self.image_size - height - top]
        return TF.pad(image, padding, fill=(124, 116, 104)), TF.pad(label, padding, fill=IGNORE_INDEX)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = self._encode_label(Image.open(label_path))
        if abs(image.size[0] / image.size[1] - label.size[0] / label.size[1]) > 0.01:
            raise RuntimeError(f"Image/label aspect-ratio mismatch for {image_path}: {image.size} vs {label.size}")
        image, label = self._resize_pair(image, label)
        if self.augment and random.random() < 0.5:
            image, label = TF.hflip(image), TF.hflip(label)
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.from_numpy(np.asarray(label, dtype=np.int64).copy()), image_path.name

    def __len__(self):
        return len(self.samples)
