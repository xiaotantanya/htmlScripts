import shutil

def copy_and_rename_file(source_path, destination_path, new_filename):
    try:
        shutil.copy(source_path, destination_path)
        
        # 构建目标文件的完整路径，包括新的文件名
        destination_file_path = shutil.os.path.join(destination_path, new_filename)
        
        # 移动并更改文件名
        shutil.move(shutil.os.path.join(destination_path, shutil.os.path.basename(source_path)), destination_file_path)
        
        print(f"文件从 {source_path} 复制到 {destination_file_path} 并更名成功！")
    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确。")
    except PermissionError:
        print("没有复制文件的权限，请检查文件和目标路径的权限。")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法
# source_file = 'pub_name.html'
# destination_folder = './'
# new_filename = 'new_file_name.txt'

# copy_and_rename_file(source_file, destination_folder, new_filename)

def main():
    source_file = 'pub_name.html'
    destination_folder = './'
    # 年份
    years = [2023,2022,2021,2020,2019,2018,2017,2016]
    # 文章名字，通过年份分类
    titles = [
        # 2023
        [
           "A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection",
           "DENet: Disentangled Embedding Network for Visible Watermark Removal",
           "Unsupervised domain adaptation using fuzzy rules and stochastic hierarchical convolutional neural networks",
           "iEnhancer-DCSA: identifying enhancers via dual-scale convolution and spatial attention"
        ],
        # 2022
        [
            "General Object Pose Transformation Network from Unpaired Data",
            "Iterative Refinement for Multi-source Visual Domain Adaptation",
            "Improving Generative Adversarial Networks with Local Coordinate Coding",
            "Visual Grounding via Accumulated Attention",
            "DF-Net:Deep Fusion Network for Multi-source Vessel Segmentation",
            "Cost-Sensitive Portfolio Selection via Deep Reinforcement Learning",
            "Self-Supervised Object Localization with Joint Graph Partition",
            "Dense Semantics-Assisted Networks For Video Action Recognition",
            "Transferable Feature Selection for Unsupervised Domain Adaptation",
            "EpNet: Power lines foreign object detection with Edge Proposal Network and data composition",
            "A Tensor-based Markov Chain Model for Heterogeneous Information Network Collective Classification",
            "Speaker extraction network with attention mechanism for speech dialogue system",
            "CAM-based non-local attention network for weakly supervised fire detection",
            "Semi-supervised Feature Selection via Structured Manifold Learning",  
        ],
        # 2021
        [
            "Self-supervised 3D Skeleton Action Representation Learning with Motion Consistency and Continuity",
            "Context Decoupling Augmentation for Weakly Supervised Semantic Segmentation",
            "Modeling the Uncertainty for Self-supervised 3D Skeleton Action Representation Learning",
            "MV-TON: Memory-based Video Virtual Try-on network",
            "Structure-aware Mathematical Expression Recognition with Sequence-Level Modeling",
            "StackRec: Efficient Training of Very Deep Sequential Recommender Models by Iterative Stacking",
            "Fast Manifold Ranking with Local Bipartite Graph",
            "Deep Level Set Learning for Optic Disc and Cup Segmentation",
            "Joint Visual and Semantic Optimization for zero-shot learning",
            "Graph Neural Network for 6D Object Pose Estimation",
            "Towards effective deep transfer via attentive feature alignment",
            "Knowledge Preserving and Distribution Alignment for Heterogeneous Domain Adaptation",
            "CycleSegNet: Object Co-Segmentation With Cycle Refinement and Region Correspondence",
            "Selection of diverse features with a diverse regularization",
            "Learning Sparse PCA with Stabilized ADMM Method on Stiefel Manifold",
            "Online Adaptive Asymmetric Active Learning with Limited Budgets",
            "Hierarchical Human-like Deep Neural Networks for Abstractive Text Summarization"
        ],
        # 2020
        [
            "LABIN: Balanced Min Cut for Large-scale Data",
            "An Ensemble of Generation- and Retrieval-Based Image Captioning With Dual Generator Generative Adversarial Network",
            "Domain-attention Conditional Wasserstein Distance for Multi-source Domain Adaptation",
            "Human interaction learning on 3D skeleton point clouds for video violence recognition",
            "Graph Edit Distance Reward: Learning to Edit Scene Graph",
            "Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis",
            "Geometric Knowledge Embedding for Unsupervised Domain Adaptation",
            "Hierarchical fusion of common sense knowledge and classifier decisions for answer selection in community question answering",
            "Fg2seq: Effectively Encoding Knowledge for End-To-End Task-Oriented Dialog"
        ],
        # 2019
        [
            "Breaking Winner-takes-all: Iterative-winners-out Networks for Weakly Supervised Temporal Action Localization",
            "Attend and Imagine: Multi-label Image Classification with Visual Attention and Recurrent Neural Networks",
            "Online Heterogeneous Transfer Learning by Knowledge Transition",
            "Subspace Weighting Co-Clustering of Gene Expression Data",
            "Pyramid Graph Networks with Connection Attentions for Region-Based One-Shot Semantic Segmentation",
            "Knowledge-enhanced Hierarchical Attention for Community Question Answering with Multi-task and Adaptive Learning",
            "From whole slide imaging to microscopy: Deep microscopy adaptation network for histopathology cancer image classification",
            "Attention Guided Network for Retinal Image Segmentation",
            "PM-NET: Pyramid Multi-Label Network for Optic Disc and Cup Segmentation",
            "Oversampling for Imbalanced Data via Optimal Transport",
            "PM-Net: Pyramid Multi-label Network for Joint Optic Disc and Cup Segmentation",
            "Auto-Embedding Generative Adversarial Networks For High Resolution Image Synthesis",
            "Guided M-Net for High-Resolution Biomedical Image Segmentation with Weak Boundaries",
            "From Whole Slide Imaging to Microscopy: Deep Microscopy Adaptation Network for Histopathology Cancer Image Classification",
            "Hierarchical human-like strategy for aspect-level sentiment classification with sentiment linguistic knowledge and reinforcement learning",
            
        ],
        # 2018
        [
            "Online Heterogeneous Transfer by Hedge Ensemble of Offline and Online Decisions",
            "Supervised Feature Selection With a Stratified Feature Weighting Method",
            "Discrimination-aware Channel Pruning for Deep Neural Networks",
            "Cartoon-to-Photo Facial Translation with Generative Adversarial Networks",
            "Adversarial Learning with Local Coordinate Coding",
            "Online Adaptive Asymmetric Active Learning for Budgeted Imbalanced Data",
            "Semi-Supervised Optimal Transport for Heterogeneous Domain Adaptation",
            "Visual Grounding via Accumulated Attention",
            "Double forward propagation for memorized batch normalization",
            "A Stratified Feature Ranking Method for Supervised Feature Selection",
            "Multi-instance transfer metric learning by weighted distribution and consistent maximum likelihood estimation"
        ],
        # 2017
        [
            "Online Transfer Learning with Multiple Homogeneous or Heterogeneous Sources",
            "MR-NTD: Manifold Regularization Nonnegative Tucker Decomposition for Tensor Data Dimension Reduction and Representation",
            "A Unified Framework for Metric Transfer Learning",
            "Online Transfer Learning by Leveraging Multiple Source Domains",
            "Multi-Instance Metric Transfer Learning for Genome-Wide Protein Function Prediction",
            "Leveraging Question Target Word Features through Semantic Relation Expansion for Answer Type Classification",
            "On the Flatness of Loss Surface for Two-layered ReLU Networks",
            "Learning Discriminative Correlation Subspace for Heterogeneous Domain Adaptation",
            "Tensor based Relations Ranking for Multi-relational Collective Classification",
            "A Self-Balanced Min-Cut Algorithm for Image Clustering",
            "Extremely Randomized Forest with Hierarchy of Multi-label Classifiers"
        ],
        # 2016
        [
            "ML-Forest: A Multi-label Tree Ensemble Method for Multi-Label Classification",
            "Online feature selection of Class Imbalance via PA algorithm",
            "Multi-Instance Multi-Label Distance Metric Learning for Genome-Wide Protein Function Prediction",
            "A fast Markov chain based algorithm for MIML learning",
            "Online Heterogeneous Transfer Learning by Weighted Offline and Online Classifiers",
            "Online Multi-Instance Multi-Label Learning for Protein Function Prediction",
            "Joint Classification with Heterogeneous labels using random walk with dynamic label propagation",
            "Individual Judgments Versus Consensus: Estimating Query-URL Relevance"
        ]
    ]
    # 论文作者，与titles中论文位置对应
    authors = [
        # 2023
        [
            ["Yukun Su","Jingliang Deng","Ruizhou Sun","Guosheng Lin*","Hanjing Su","Qingyao Wu*"],
            ["Ruizhou Sun#","Yukun Su#","Qingyao Wu*",]
        ],
        # 2022
        [
            
        ],
        # 2021
        [
            
        ],
        # 2020
        [
            
        ],
        # 2019
        [
            
        ],
        # 2018
        [
            
        ],
        # 2017
        [
            
        ],
        # 2016
        [
            
        ]
    ]
    messages = {}
    for year in years:
        messages[year] = []

    