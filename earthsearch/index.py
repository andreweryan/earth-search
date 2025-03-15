import os
import torch
from typing import Literal, List
from .core import ImageSimilaritySearch

def index(
        image_dir: str,
        index_path: str,
        indexed_images_path: str,
        index_type: str = "L2",
        model_type: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", 
                                "dinov2_vits14", "dinov2_vits14_reg", "dinov2_vitb14", "dinov2_vitb14_reg", 
                                "dinov2_vitl14", "dinov2_vitl14_reg", "dinov2_vitg14", "dinov2_vitg14_reg"] = "dinov2_vits14_reg",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
        batch_size: int = 32,
        overwrite_index: bool = False
            ) -> None:
    """
    
    Args:
        image_dir (str): Path to directory of chipped images to index
        index_path (str): Path to saved index 
        indexed_images_path (str): Path to file containing indexed image paths
        valid_exts (List[str]): Image extensions to filter for
        index_type (str): Type of index to use (currently only using L2)
        model_type (str): Type of model to use
        device (str): Device to use for feature extraction ("cuda", "mps", "cpu")
        batch_size (int): Number of images per batch
        overwrite_index (bool): Overwrite existing index of same index_path

    Return:
        None
    """
    if "_reg" in model_type and torch.mps.is_available():
        print("Some PyTorch functionality using DINOv2 models with Registers "
        "not supported by MPS yet, reverting to CPU")
        device = "cpu"
    else:
        device = device if device else "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    
    if not (os.path.exists(index_path) and os.path.exists(indexed_images_path)) or overwrite_index:
        print(f"Indexing Images")
        indexer = ImageSimilaritySearch(model_type=model_type, index_type=index_type, device=device)
        indexer.index_images(image_dir, batch_size=batch_size)
        indexer.save(index_path, indexed_images_path) # Save the index and image paths file
    else:
        print(f"Index already exists...")

    return None

if __name__ == "__main__":
    index()