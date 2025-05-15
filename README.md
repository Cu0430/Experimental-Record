# Progressive Hierarchical Alignment for Few-Shot Cross-Modal Remote Sensing Image Classification
本仓库用于记录本人在科研实习中的实验过程与模型演化：**基于渐进式层次对齐的小样本跨模态遥感图像分类研究**。每个子文件夹（如 v1/, v2/, ...）代表模型架构的一次重要迭代，均配有相应的 README.md 文件详细说明改动与实验结果。该实验代码从2025.4.1开始进行编写和调整。
# 研究动机
在小样本多光谱（MS）与全色（PAN）遥感图像联合分类任务中，面临两个关键挑战：  
1）多层次语义错位：不同模态在浅层与深层存在语义空间不一致；  
2）模态特异性信息丢失：过度对齐可能损害模态特有的判别能力。  
为解决上述问题，本研究设计了一种渐进式层次对齐机制，并提出了模态补偿字典模块，以提升特征的判别性和鲁棒性。
# 当前结果
实验中使用的数据集是呼和浩特的MS和PAN图像。整个实验过程经历了过山车式的变化：一开始AA只有40%-50%，调整网络结构之后，AA能到71%。后来发现有原则性的数据处理错误，调整之后AA又回到了30%左右。之后又发现了数据量的问题，又调整了超参数、网络结构等，目前取得的最佳结果如下：  
**Overall Accuracy (OA): 73.37%  
Average Accuracy (AA): 75.18%  
Kappa Coefficient: 69.88%**  
与当前 SOTA 方法相比，各指标约存在 **2% 左右的差距**，具有进一步优化的潜力。
# Ps
My thesis supervisor is Zhu Hao, associate professor of Xidian University. He has some achievements in deep learning and remote sensing image processing. He supervises undergraduate research and publishes papers in IEEE TGRS journals(IF:7.5, 中科院一区TOP期刊). This is his personal homepage at https://faculty.xidian.edu.cn/ZHUHAO/zh_CN/index.htm
