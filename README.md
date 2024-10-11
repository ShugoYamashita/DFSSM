# DFSSM (ACCV 2024)
This is the official code of the ACCV 2024 paper [Image Deraining with Frequency-Enhanced State Space Model](https://arxiv.org/abs/2405.16470).

## Abstract
Removing rain degradations in images is recognized as a significant issue. In this field, deep learning-based approaches, such as Convolutional Neural Networks (CNNs) and Transformers, have succeeded. Recently, State Space Models (SSMs) have exhibited superior performance across various tasks in both natural language processing and image processing due to their ability to model long-range dependencies. This study introduces SSM to image deraining with deraining-specific enhancements and proposes a Deraining Frequency-Enhanced State Space Model (DFSSM). To effectively remove rain streaks, which produce high-intensity frequency components in specific directions, we employ frequency domain processing concurrently with SSM. Additionally, we develop a novel mixed-scale gated-convolutional block, which uses convolutions with multiple kernel sizes to capture various scale degradations effectively and integrates a gating mechanism to manage the flow of information. Finally, experiments on synthetic and real-world rainy image datasets show that our method surpasses state-of-the-art methods. Code is available at https://github.com/ShugoYamashita/DFSSM.

## Citation
If you use this code or models in your research and find it helpful, please cite the following paper:

```
@inproceedings{yamashita2024image,
    title={Image Deraining with Frequency-Enhanced State Space Model},
    author={Yamashita, Shugo and Ikehara, Masaaki},
    booktitle={ACCV},
    year={2024}
}
```
