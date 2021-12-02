# Egyptian Hieroglyph Classification using Capsule Network

This is a [PyTorch](https://pytorch.org/) implementation of the paper "Egyptian Hieroglyph Classification using Capsule Network" by Daniel Jung, Kangdong Yuan, and Kaamran Raahemifar.

Please download the dataset "EgyptianHieroglyphDataset_Original" (40 classes) at my [Google Drive](https://drive.google.com/drive/folders/1bhnMJ8NbCa-qw53EKy-olZp3cJKZU_jc?usp=sharing).<br />
Please download the dataset "EgyptianHieroglyphDataset_Original_Clean" (40 classes) at my [Google Drive](https://drive.google.com/drive/folders/1X5HdFvgWJOVtA-GxBLr1K_0FHJS2RZcZ?usp=sharing).<br />
Please download the dataset "EgyptianHieroglyphDataset_Original_134" (134 classes) at my [Google Drive](https://drive.google.com/drive/folders/1-qaDjJpZv84XIXYX1yAZnqUQBKAMxsgu?usp=sharing).

**Image** | ![alt text](/example/D21.png) | ![alt text](/example/E34.png) | ![alt text](/example/V31.png) 
------------ | ------------ | ------------- | -------------
**Gardener Label** | D21 | E34 | V31

Steps for running <b>full Python code</b>:
1. Download "EgyptianHieroglyphDataset_Original" dataset from my Google Drive
2. Download "src" folder in this repo
3. Install all the requirements for Python packages
4. Run main.py
5. The training will start right away!

Steps for running <b>Jupyter Notebook</b>:
1. Choose any folder of model architectures that you want to run
2. Click one of the ipynb files in my repo (same model with different cases whether pretrained and whether clean dataset)
3. Click "Open in Colab"
4. Download "EgyptianHieroglyphDataset_Original" dataset from my Google Drive and store it into your Google Drive
5. Connect the Google Colab with your Google Drive and run the codes
6. The training will start right away!

**Performances (train with data augmentation except CapsNet and test with data augmentation except CapsNet):**
**Model** | Pretrained | From scratch | Pretrained with clean dataset | From scratch with clean dataset
------------ | ------------ | ------------- | ------------- | -------------
ResNet-50 | 98.6% | 97.0% | 99.36% | 95.43%
Inception-v3 | **99.2%** | **98.2%** | **99.4%** | 98.4% 
Xception | 98.74% | 97.48% | 99.21% | 97.9%
Glyphnet | Not yet | Not yet | Not yet | Not yet | 
Capsule Network | - | 98.11% | - | **98.74%**
Capsule Network with data augmentation | - | 93.07% | - | 93.37%

Our data augmentation techniques include:
1. Random Horizontal Flip
2. Random Rotation (degrees between -10 and 10)
3. Random Affine (Random movement of entire image)
4. Random Color Jitter
5. Rnadom Erasing (of a small part of each image)

**Performances (train without data augmentation and test with data augmentation):**
**Model** | Pretrained | From scratch | Pretrained with clean dataset | From scratch with clean dataset
------------ | ------------ | ------------- | ------------- | -------------
ResNet-50 | 52.44% | 22.36% | 58.67% | 22.34%
Inception-v3 | **78.58%** | **64.25%** | **79.02%** | **61.35%**
Xception | 49.76% | 46.61% | 48.26% | 42.90%
Glyphnet | - | Not yet | - | Not yet | 
Capsule Network | - | 49.60% | - | 48.89%

**Performances (40 classes with weighted sampler - 50 epochs):**
**Model** | Pretrained | From scratch | 
------------ | ------------ | ------------- | 
ResNet-50 | 99.21% | 97.16% | 
Inception-v3 | **99.52%** | 97.00% |
Xception | 98.58% | **97.47%** |
Glyphnet | - |  |
Capsule Network | - | 91.32% |

Weighted sampler allows our data loader to load similar number of images for each label. This prevent wrong performance evaluation scheme with too much emphasis on image labels with too many images (our dataset is very unbalanced). For each label, our model is tested with 10 ~ 20 images.

**Performances (40 classes with weighted sampler for Ensemble Learning):**
**Model** | Pretrained | From scratch | # of models | # of epochs |
------------ | ------------ | ------------- | ------------- | ------------- |
ResNet-50 | 99.37% | 96.53% | 3 | 30 |
Inception-v3 |  |  |  |  |
Xception |  |  |  |  |
Glyphnet | - |  | | |
Capsule Network | - |  | | |

**Performances (134 classes):**
**Model** | Pretrained | From scratch | 
------------ | ------------ | ------------- 
ResNet-50 | 91.56% | 87.57% | 
Inception-v3 | **94.49%** | **91.79%** | 
Xception | 92.08% | 88.98% | 
Glyphnet | - |  | 
Capsule Network | - | CUDA out of memory | 

We have left any method that gives CUDA out of memory in Google Colab Pro as we believe our interdisciplinary models should be used only when the Egyptologists are accessible without any huge GPU server in their disposal.

**Performances (134 classes with weighted sampler on both training and testing):**
**Model** | Pretrained | From scratch | 
------------ | ------------ | ------------- 
ResNet-50 | 76.90% | 65.06% | 
Inception-v3 | **80.77%** | 70.34% | 
Xception | 80.42% | **72.80%** | 
Glyphnet | - |  | 
Capsule Network | - | CUDA out of memory | 

**Performances (134 classes with weighted sampler for Ensemble Learning):**
**Model** | Pretrained | From scratch | # of models | # of epochs |
------------ | ------------ | ------------- | ------------- | ------------- |
ResNet-50 | **78.54%** |  | 3 | 10 |
Inception-v3 | 73.58% |  | 3 | 10 |
Xception | 78.19% |  | 3 | 20 |
Glyphnet | - |  | | |
Capsule Network | - |  | | |


Pretrained performance for Capsule Network is not on this paper for both computational limit and our pursuit on training from scratch approach using Capsule Network

**Capsule Network Hyperparameter tuning**
**Model** | num_capsules | routing_iterations | performance
------------ | ------------ | ------------- | -------------
Capsule Network | 8 | 3 | **98.74%**
Capsule Network | 12 | 3 | 97%
Capsule Network | 15 | 3 | CUDA Out of Memory

Prior implementations:
1. [GlyphReader by Morris Franken](https://github.com/morrisfranken/glyphreader) which extracts features using Inception-v3 and classifies hieroglyphs using SVM.

TODO:
1. Implementation for Glyphnet
2. Implement a larger Capsule Network with more capsules and dynamic routings (CUDA Out of Memory)
3. Implementation of ROC curve
4. Trian on 171 classes and test on 40 classes
5. Ensemble CNN model
6. Bagging and Boosting with CNN models
