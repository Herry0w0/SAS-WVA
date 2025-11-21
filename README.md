This repository contains the official PyTorch implementation of the paper:
**SAS-WVA: Semantically-Aware Seeding and Weighted Voronoi Assignment for 3D Point Cloud Oversegmentation**.

## 🛠️ Installation

The code has been tested with **Python 3.10**, **PyTorch 2.3.1**, and **CUDA 12.1**.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Herry0w0/SAS-WVA.git
    cd SAS-WVA
    ```

2.  **Create the Conda environment:**
    We provide a ready-to-use `env.yaml` file.

    ```bash
    conda env create -f env.yaml
    conda activate overseg
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
    MAX_JOBS=8 pip install flash-attn --no-build-isolation
    ```

## 📂 Data Preparation (S3DIS)

Please organize the point cloud data for all Areas into `.npy` files.

1.  **Data Format:** Each `.npy` file must be a matrix of shape `(N, 10)`, containing the following channels in order:

      * **Columns 0-2:** `x, y, z` (Coordinates)
      * **Columns 3-5:** `r, g, b` (Color)
      * **Columns 6-8:** `nx, ny, nz` (Normal Vectors)
      * **Column 9:** `label` (Semantic Label)

    > **Note:** You need to pre-compute the normal vectors for each point using an external algorithm (e.g., PCA or Open3D) before saving them into the npy files.

2.  **Configuration:**
    After preparing the data, modify the `root` parameter in `config/s3dis.yaml` to point to the directory containing your `.npy` files:

    ```yaml
    data:
      root: /path/to/your/S3DIS_npy_data
    ```

## 🚀 Training

We provide a shell script to launch distributed training.

To train the model (default uses S3DIS Area 5 for validation):

```bash
bash scripts/train_s3dis.sh
```

## 🙏 Acknowledgements

We would like to thank the authors of [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) for their excellent work. Parts of our code are derived from their open-source repository.