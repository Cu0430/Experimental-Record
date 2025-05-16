# Progressive Hierarchical Alignment for Few-Shot Cross-Modal Remote Sensing Image Classification
This repository is used to record the experimental process and model evolution in my research internship: **a study on cross-modal remote sensing image classification for small samples based on progressive hierarchical alignment.** Each subfolder (e.g., v1/, v2/, ...) represents an important iteration of the model architecture, and is accompanied by a corresponding README.md file detailing the changes and experimental results. The experimental code was written and adapted from 2025.3.22
# Research Motivation
In the task of joint classification of small sample multispectral (MS) and panchromatic (PAN) remote sensing images, two key challenges are faced:  
1) Multi-level semantic misalignment: shallow feature misalignment errors are transmitted layer by layer in the network, resulting in the inability to achieve true cross-modal consistency by aligning only high-level semantic features;
2) Loss of modality-specific information: over-alignment may impair modality-specific discriminative capabilities.

To solve the above problems, this study designs **a progressive hierarchical alignment mechanism** and proposes **a modality-compensated dictionary module** to improve the discriminative and robustness of features.
# Current Results
The datasets used in the experiment are MS and PAN images of Hohhot. The whole experimental process experienced **a roller coaster**: at the beginning, the AA was only 40%-50%, and after adjusting the network structure, the AA could go up to 71%. Then it was found that there was a principle data processing error, and after the adjustment, the AA went back to about 30%. After that, the problem of data volume was found again, and the hyper-parameters and network structure were adjusted, etc. The best results achieved so far are as follows:\
**Overall Accuracy (OA): 73.37%  
Average Accuracy (AA): 75.18%  
Kappa Coefficient: 69.88%**  
Compared with the current SOTA method, there is a gap of about 2% in each index, which has the potential for further optimization.
# Ps
My thesis supervisor is Zhu Hao, associate professor of Xidian University. He has some achievements in deep learning and remote sensing image processing. He supervises undergraduate research and publishes papers in IEEE TGRS journals(IF:7.5, Chinese Academy of Sciences Region I TOP journals). This is his personal homepage at https://faculty.xidian.edu.cn/ZHUHAO/zh_CN/index.htm
