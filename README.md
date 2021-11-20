# Egyptian_Hieroglyph_Classification_CapsNet_

This is a [PyTorch](https://pytorch.org/) implementation of the paper ["A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification" by Barucci et al](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9528382&tag=1).

Please download the dataset "EgyptianHieroglyphDataset_Original" at my [Google Drive](https://drive.google.com/drive/folders/1bhnMJ8NbCa-qw53EKy-olZp3cJKZU_jc?usp=sharing).<br />
Please download the dataset "EgyptianHieroglyphDataset_Original_Clean" at my [Google Drive](https://drive.google.com/drive/folders/1X5HdFvgWJOVtA-GxBLr1K_0FHJS2RZcZ?usp=sharing).

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

**Model** | Pretrained | From scratch | Pretrained with clean dataset | From scratch with clean dataset
------------ | ------------ | ------------- | ------------- | -------------
ResNet-50 | 98.6% | 97.0% | 99.36% | Not yet
Inception-v3 | 99.2% | 98.2% | 99.4% | 98.4% 
Xception | 98.74% | 97.48% | Not yet | 97.9%    
Capsule Network | - | 98.11% | - | 98.74%

Prior implementations:
1. [GlyphReader by Morris Franken](https://github.com/morrisfranken/glyphreader) which extracts features using Inception-v3 and classifies hieroglyphs using SVM.

TODO:
1. Implementation for Glyphnet
2. Implementation for Xception
3. Implement a larger Capsule Network with more capsules and dynamic routings
4. Test on horizontal and vertical flip testing (with no horizontal and vertical flip training)
