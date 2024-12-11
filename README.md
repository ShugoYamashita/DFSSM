# DFSSM (ACCV 2024)
This is the official code of the ACCV 2024 paper [Image Deraining with Frequency-Enhanced State Space Model](https://arxiv.org/abs/2405.16470).

Visual results are provided, and the detailed code will be released soon.

## Abstract
Removing rain degradations in images is recognized as a significant issue. In this field, deep learning-based approaches, such as Convolutional Neural Networks (CNNs) and Transformers, have succeeded. Recently, State Space Models (SSMs) have exhibited superior performance across various tasks in both natural language processing and image processing due to their ability to model long-range dependencies. This study introduces SSM to image deraining with deraining-specific enhancements and proposes a Deraining Frequency-Enhanced State Space Model (DFSSM). To effectively remove rain streaks, which produce high-intensity frequency components in specific directions, we employ frequency domain processing concurrently with SSM. Additionally, we develop a novel mixed-scale gated-convolutional block, which uses convolutions with multiple kernel sizes to capture various scale degradations effectively and integrates a gating mechanism to manage the flow of information. Finally, experiments on synthetic and real-world rainy image datasets show that our method surpasses state-of-the-art methods. Code is available at https://github.com/ShugoYamashita/DFSSM.

## Evaluations
1) *for Rain200H/L, SPA-Data, and LHP-Rain datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/swz30/Restormer/blob/main/Deraining/evaluate_PSNR_SSIM.m).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation).

| Synthetic or Real | Dataset | PSNR | SSIM | Visual Results |
|-------------------|---------|------|------|----------------|
| Synthetic | Rain200H | 32.99 | 0.9403 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |
| Synthetic | Rain200L | 41.81 | 0.9905 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |
| Synthetic | DID-Data | 35.66 | 0.9671 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |
| Synthetic | DDN-Data | 34.53 | 0.9608 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |
| Real | SPA-Data | 49.55 | 0.9939 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |
| Real | LHP-Rain | 33.99 | 0.9322 | [Download](https://drive.google.com/drive/folders/1LsdsVkCgNhPpN5-8njaC39eA6sKPxDVC?usp=sharing) |

## Citation
If you use this code or models in your research and find it helpful, please cite the following paper:

```
@InProceedings{Yamashita_2024_ACCV,
    author    = {Yamashita, Shugo and Ikehara, Masaaki},
    title     = {Image Deraining with Frequency-Enhanced State Space Model},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {3655-3671}
}
```
