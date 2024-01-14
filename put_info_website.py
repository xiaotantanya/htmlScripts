from bs4 import BeautifulSoup


def write_a_file(title_information, author_information):
    '''
    title_information
        type dict
        title_name string
        abstract string
        cite string
    author_information
        type dict
        authors List[{"name":xxx, "index": yyy}]
    '''
    with open("pub_name.html","r",encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    head_title = soup.head.title
    head_title.string = title_information["title_name"]
    another_title = soup.body.section.div.div.div.div.h1
    another_title.string = title_information["title_name"]
    author_information_div = soup.body.section.div.div.div.div.div
    authors = author_information["authors"]
    for author in authors:
        author_span = soup.new_tag('span',attrs={'class': 'author-block'})
        author_link = soup.new_tag('a', href=' ')
        author_link.string = author["name"]
        sup_tag = soup.new_tag('sup')
        sup_tag.string = author["index"]
        author_span.append(author_link)
        author_link.append(sup_tag)
        author_information_div.append(author_span)
    
    abstract = soup.body.section.div.div.div.div.p
    abstract.string = title_information["abstract"]
    
    cite_code = soup.body.section.div.pre.code
    cite_code.string = title_information["cite"]
    with open("{}_{}.html".format(title_information["year"],title_information["index"]),"w",encoding="utf-8") as file:
        file.write(soup.prettify())

def write_files(title_informations, author_informations):
    title_info_num = len(title_informations)
    author_info_num = len(author_informations)
    assert title_info_num == author_info_num, "Need Equal!"
    for i in range(0, title_info_num):
        title_information = title_informations[i]
        author_information = author_informations[i]
        write_a_file(title_information, author_information)
        

def read_title_information(title_file, abstract_file, cite_file):
    name_dict = read_name_information(title_file)
    abstract_dict = read_abstract_information(abstract_file)
    cite_dict = read_cite_information(cite_file)
    title_file = dict()
    year_keys = name_dict.keys()
    abstract_keys = abstract_dict.keys()
    cite_dict_keys = cite_dict.keys()
    
    assert year_keys == abstract_keys
    assert abstract_keys == cite_dict_keys
    
    for year in year_keys:
        name_list = name_dict[year]
        abstract_list = abstract_dict[year]
        cite_list = cite_dict[year]
        assert name_list == abstract_list
        assert abstract_list == cite_list
        title_file[year] = list()
        for i in len(name_list):
            title_file[year][i] = dict()
            title_file[year][i]["title_name"] = name_list[i]
            title_file[year][i]["abstract"] = abstract_list[i]
            title_file[year][i]["cite"] = cite_list[i]
    
    return title_file

def read_author_information(author_info_yml):
    import yaml
    with open(author_info_yml,'r') as file:
        author_info_dict = yaml.safe_load(file)
        return author_info_dict

def read_name_information(title_file):
    titles = {
        "2023":
        [
           "A Unified Transformer Framework for Group-based Segmentation: Co-Segmentation, Co-Saliency Detection and Video Salient Object Detection",
           "DENet: Disentangled Embedding Network for Visible Watermark Removal",
           "Unsupervised domain adaptation using fuzzy rules and stochastic hierarchical convolutional neural networks",
           "iEnhancer-DCSA: identifying enhancers via dual-scale convolution and spatial attention"
        ],
        "2022":
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
        "2021":
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
        "2020":
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
        "2019":
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
        "2018":
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
        "2017":
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
        "2016":
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
    }
    return titles

def read_abstract_information(abstract_file):
    import re
    abstract_dict = dict()
    year = 0
    with open(abstract_file,"r",encoding="utf-8") as file:
        content = file.readline().rstrip('\n')
        while(content != ''):
            if(len(content) == 4):
                assert content.isdigit() == True
                year = int(content)
                abstract_dict[year] = list()
                content = file.readline().rstrip('\n')
                continue
            else:
                content = re.sub(r'^\d+\.','',content)
                abstract_dict[year].append(content)
                content = file.readline().rstrip('\n')

    return abstract_dict

def read_cite_information(cite_file):
    import re
    cite_dict = dict()
    year = 0
    total_string = ""
    with open(cite_file,"r",encoding="utf-8") as file:
        content = file.readline().rstrip('\n')
        while(content != ''):
            if(len(content) == 4):
                assert content.isdigit() == True
                year = int(content)
                cite_dict[year] = list()
                content = file.readline().rstrip('\n')
                continue
            else:
                pattern = re.compile(r'^\d+\.')
                if(pattern.match(content)):
                    if total_string != '':
                        cite_dict[year].append(total_string)
                    total_string = ""
                    content = re.sub(r'^\d+\.','',content)
                    total_string = total_string + content + '\n'
                    content = file.readline().rstrip('\n')
                else:
                    content = content.strip()
                    total_string = total_string + "  " + content + '\n'
                    content = file.readline().rstrip('\n')
    return cite_dict
# print(soup)

# original_html = '<html><body><p>Existing content</p></body></html>'
# new_block = '<div><p>New content</p></div>'

# soup = BeautifulSoup(original_html, 'html.parser')
# new_block_soup = BeautifulSoup(new_block, 'html.parser')

# soup.body.append(new_block_soup.div)

# print(soup.prettify())

if __name__ == '__main__':
    # read_author_information("./institute.yml")
    # read_abstract_information("./abstract.txt")
    read_cite_information("./citation.txt")