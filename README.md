# TmPred
TmPred aims to accurately predict the melting temperature (Tm) of thermophilic proteins.

The TmPred model consists of three modules: a Graph Convolutional Network (GCN) module, an attention (Graphormer) module, and a Fully Connected (FC) layer. The model input includes node and edge information. Node information is derived from the amino acid sequence of the protein, with each residue's amino acid being represented by a specific-dimensional embedding vector generated through a protein language model. Edge information is derived from the protein’s structure. The residue contact map, which is calculated to reflect the contact relationships between each amino acid residue and others in the protein, serves as the input for the edge information.

![Fig1](https://github.com/user-attachments/assets/ed048915-6806-426c-b25c-64087031cb9f)


