## Requirements
* python>=3.8
* SimpleITK==2.2.1
* mmcv==2.2.0
* numpy==1.24.3

## Synthetic Tumor Generation Method Based on Medical Prior Knowledge
In liver tumor segmentation tasks, the scarcity of annotated data (e.g., only 131 cases in the LiTS dataset) and the insufficient samples of early small tumors pose significant challenges. Traditional generative models (such as GANs or VAEs) tend to produce blurry boundaries or distorted shapes. This project innovatively proposes a procedural synthetic tumor generation framework that incorporates medical prior knowledge from the Liver Imaging Reporting and Data System (LI-RADS) to achieve high-fidelity tumor synthesis. This approach effectively alleviates the impact of data scarcity on model training.

## Data
* [LiTS](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation) 131 CT scans for segmentation of the liver as well as tumor lesions.

# Date process

### 1. Intelligent Localization and Vessel Avoidance
- **Vessel Segmentation**: The `segment_vessels` function quickly locates vascular regions using a preset HU value range (150–300).
- **Safety collision Detection**: The `gen_position` algorithm, combined with distance transformation (`distance_transform_edt`), ensures that the tumor generation points maintain anatomically reasonable distances from blood vessels.
- **Liver Mask Erosion**: A 5x5 kernel is used for edge erosion to prevent tumors from being generated at the liver boundary.

### 2. Multi-Stage Morphological Modeling
- **Basic Geometric Generation**: The `get_ellipsoid` function constructs a 3D ellipsoid to simulate the regular shape of early-stage tumors.
- **Composite Deformation Strategy**:
  - **Elastic Deformation**: `apply_complex_deformation` dynamically adjusts deformation intensity based on tumor diameter (D) with σ_e = 0.5 + 0.1 * (D/30)².
  - **Fractal Noise**: For tumors ≥10mm, Perlin noise (λ = 0.1 * (1 + D/20)) is applied to generate lobulated edges.
- **Morphological Optimization**: `binary_erosion` and `binary_dilation` are used to maintain the anatomical relationship between the tumor and liver parenchyma.

### 3. Pathological Texture Synthesis
- **HU Value Modeling**:
  - **Necrotic Core**: Increases the HU value by 30 in the central region (`necrosis_mask`).
  - **Steatosis**: Random patches attenuate HU values by 10–30 (`fat_mask`).
- **Gaussian Smoothing**: A Gaussian filter with σ = 1.5 is applied to eliminate artificial traces.
- **Edge Enhancement**: Morphological edge detection enhances the capsule effect (`edge_mask * 25`).

<p align="center"><img width="100%" src="figures/syn.png" /></p>

## Run the codes
```bash
gen_synthesis_tumor: python main.py -i input_dir -o output_dir
