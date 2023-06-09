# MTDLGC
ConvGNNs have been widely applied in traffic forecasting applications. How to select a suitable spatial weighting scheme for ConvGNNs, however, remains challenging, given the complicated spatial dependencies of traffic variables in real road networks. In this study, we propose a novel ConvGNN, termed learnable graph convolutional (LGC) network, which learns spatial weightings between a road and its k-hop neighbors as learnable parameters in the spatial convolutional operator. A dynamic LGC (DLGC) network is further proposed to learn the dynamics of spatial weightings by explicitly considering the temporal correlations of spatial weightings at different times of the day. A multi-temporal DLGC (MT-DLGC) network is developed for end-to-end forecasting of traffic variables in road networks. 
![MT-LGC2](https://github.com/Mayaohong/MTDLGC/assets/136045955/2e84d832-82d7-45aa-990b-ee40bb6ef3f7)

# Installation
MTDLGC can be installed from source code.
Please execute the following command to get the source code.

```bash
git clone https://github.com/Mayaohong/MTDLGC
cd MTDLGC
```
# Quick-Start
Before run our models, please make sure you download related datasets (PeMSD4 and PeMSD8) and put them in corresponding folder ```./raw_data/```. The download link of PeMSD4 and PeMSD8 datasets is [Google Drive](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe). This two datasets used in MTDLGC have been processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format. The further details about datasets can be found in the [doucument link](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html).

The script ```run_model.py``` is used for training and evaluating our proposed models and baselines in MTDLGC. You run the ```run_model.py``` by the following way:
```bash
python run_model.py
```
# Acknowledgments
"We would like to acknowledge the implementation framework for traffic prediction models provided by [LibCity](https://github.com/LibCity/Bigscity-LibCity). The framework served as a valuable resource and reference for our research work.  We are very grateful to the LibCity team for their contributions and efforts in advancing the field of traffic prediction."
