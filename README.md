# MMD-DTA
MMD-DTA： A multi-modal deep learning framework for drug-target binding affinity and binding region prediction

###Citation：
If you have used MMD-DTA or its modules in your research, please cite this paper

## 1. Environment setup
  python:3.7.6  

  torch:1.4.0  

  scikit-learn:0.22.1  

  pandas:1.0.1  

  numpy:1.18.1

##2.Construct a dataset
   Data preprocessing:
   The Davis dataset and Kiba dataset need to be downloaded before use，Please download the Davis and Kiba datasets yourself and place them in the Dataset folder，and run create_data.py
   The following are the sources of Davis and Kiba:
   Davis: The comparative toxicogenomics database: update 2013. Nucleic acids research, 41(D1), D1104-D1114.
   kiba:Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. Journal of Chemical Information and Modeling, 54(3), 735-743.
 


##3.start:
  python training.py  # BG-DTI
