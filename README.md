# Earth Search

A library for efficient indexing & similarity search of satellite images. 

Earth Search is built on top of Faiss (Facebook AI Similarity Search) with a number of feature extractors for indexing and querying satellite image embeddings. 

<p align="center">
  <img src="assets/queries.png">
</p>

Task List
- [ ] Update README with installation, usage, etc.
- [ ] Everything else...
- [ ] Extract location from query image filenames, export to geospatial file formats
- [ ] Integrate other feature extraction methods
- [ ] 

## Installation:


## Usage:

Package import usage:
```python
from earthsearch.chip import chip
from earthsearch.index import index
from earthsearch.search import search
from earthsearch.core import show_search_results

image_dir = "directory/path/to/satellite/image/scenes"
chip_dir = "directory/path/to/write/image/chips"
window_size = 512
stride = 0.0
valid_exts = ["tif"]

index_path = "path/to/save/index/to"
indexed_images_path = "path/to/write/indexed/image/paths/to"
index_type = "L2"
model_type = "dinov2_vits14_reg"
device = "cuda" # or "mps", "cpu
batch_size = 32

query_image = "path/to/query/for/top_k/similar/images", 
top_k = 10

chip(image_dir, chip_dir, window_size, stride, valid_exts, multiprocess=True)
index(chip_dir, index_path, indexed_images_path, index_type, model_type, device, batch_size, overwrite_index=False)
results = search(query_image, index_path, indexed_images_path, index_type, top_k, model_type, device)

for idx, result in enumerate(results):
    print(f"{idx + 1}: {result["path"]} - Distance: {result["distance"]}")
show_search_results(query_image, results, max_display=top_k)

```

CLI usage:
`earthsearch {chip,index,search} ...`

```
usage: earthsearch chip [-h] --image_dir IMAGE_DIR --chip_dir CHIP_DIR
                        [--window_size WINDOW_SIZE] [--stride STRIDE]
                        [--valid_exts [VALID_EXTS ...]] [--multiprocess]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to images
  --chip_dir CHIP_DIR   Directory path to write chips to
  --window_size WINDOW_SIZE
                        Size of sliding window, e.g., 512
  --stride STRIDE       Amount of overlap in x, y direction, e.g., 0.2
  --valid_exts [VALID_EXTS ...]
                        Image extensions to filter for
  --multiprocess        Use multiprocessing vs multithreading
```

```
usage: earthsearch index [-h] --image_dir IMAGE_DIR --index_path INDEX_PATH
                         --indexed_images_path INDEXED_IMAGES_PATH --index_type INDEX_TYPE
                         --model_type MODEL_TYPE [--device DEVICE] [--batch_size BATCH_SIZE]
                         [--overwrite_index]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to directory of images to index
  --index_path INDEX_PATH
                        Path to save index
  --indexed_images_path INDEXED_IMAGES_PATH
                        Path to save file containing indexed image paths
  --index_type INDEX_TYPE
                        Type of index to use (currently only using L2)
  --model_type MODEL_TYPE
                        Type of model to use
  --device DEVICE       Device to use for feature extraction (cuda, mps, cpu)
  --batch_size BATCH_SIZE
                        Number of images per batch
  --overwrite_index     Overwrite existing index of same index_path
(.venv)  ✝  ~/repos/earthsearch   main 
```

```
usage: earthsearch search [-h] --query_image QUERY_IMAGE --index_path INDEX_PATH
                          --indexed_images_path INDEXED_IMAGES_PATH --index_type INDEX_TYPE
                          --top_k TOP_K --model_type MODEL_TYPE [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --query_image QUERY_IMAGE
                        Path to directory of images to index
  --index_path INDEX_PATH
                        Path to saved index
  --indexed_images_path INDEXED_IMAGES_PATH
                        Path to file containing indexed image paths
  --index_type INDEX_TYPE
                        Type of index to use (currently only using L2)
  --top_k TOP_K         top k-Nearest Neighbors to return
  --model_type MODEL_TYPE
                        Type of model to use
  --device DEVICE       Device to use for feature extraction (cuda, mps, cpu)
```