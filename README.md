# TmPred
TmPred aims to accurately predict the melting temperature (Tm) of thermophilic proteins.

The TmPred model consists of three modules: a Graph Convolutional Network (GCN) module, an attention (Graphormer) module, and a Fully Connected (FC) layer. The model input includes node and edge information. Node information is derived from the amino acid sequence of the protein, with each residue's amino acid being represented by a specific-dimensional embedding vector generated through a protein language model. Edge information is derived from the protein’s structure. The residue contact map, which is calculated to reflect the contact relationships between each amino acid residue and others in the protein, serves as the input for the edge information.

![Fig1](https://github.com/user-attachments/assets/ed048915-6806-426c-b25c-64087031cb9f)

Compared to existing models for predicting the melting temperature of thermophilic proteins, TmPred demonstrates substantial improvements. When evaluated on unseen thermophilic proteins, TmPred still outperforms these models, demonstrating strong generalization capability.

# Dependencies
· python=3.11.9

· pytorch=2.4.1

· numpy=1.26.4

· pandas=2.2.2

· scipy=1.13.1

· scikit-learn=1.5.1

· transformers=4.44.1

· networkx=3.3

· multiprocess=0.70.15

· tqdm=4.66.5

# Reference
Coming soon...
