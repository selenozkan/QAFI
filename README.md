<h1 style="font-size: 2.5em;">QAFI: A Novel Method for Quantitative Estimation of Missense Variant Impact Using Protein-Specific Predictors and Ensemble Learning</h1>



---

<p style="font-size: 1.2em;">
    QAFI is a prediction method that is designed for the quantitative prediction of the functional impact of missense variants. </p>
<p style="font-size: 1.2em;">    QAFI combines predictions from ten protein-specific predictors (PSP) where each PSP is separately trained on deep mutational scanning assays with 14 sequence-based and structure-based features, using multiple linear regression models. Each PSP's prediction is then combined by taking their medians to come up with the final quantitative prediction for the variants of the tested protein. </p>

---
# Development of Protein-Specific Predictors and Their Cross-Validated Performances

The notebook `ProteinSpecificPredictors.ipynb` includes all analyses related to the development of protein-specific predictors (PSP) and the selection of 10 PSPs for the final QAFI model:

- **Exploring the Distribution**: Analysis of 30 different deep mutational scanning assays used in this study.
- **Feature Importance Analysis**: Conducted with Lasso models.
- **Cross-Validation**: Leave-one-position-out cross-validated performance for each PSP.
- **Cross Predictions**: Each PSP is used to predict the remaining 29 DMS assays separately.
- **Top 10 PSP Selection**: PSPs are ranked based on their predictive performances on remaining proteins. The top 10 consistently powerful PSPs are selected for the final QAFI method.
- **Performance Comparison**: Evaluation of the performance between each PSP and QAFI predictions.

The normalized DMS assays and the features used can be found in `data/Dataset_30proteins_features.csv`

---
# Obtaining Features for a Protein of Interest

The notebook `Features.ipynb` includes a pipeline for obtaining all 14 features step-by-step for a given protein of interest. All models developed in this study incorporate 14 features: 5 sequence-based and 9 structure-based.


## Sequence-Based Features:
- Blosum62
- PSSM
- Shannon's Entropy
- Shannon's Entropy of Sequence Neighbours
- neco

## Structure-Based Features:
- pLDDT
- pLDDT Bin
- colasi
- Fraction Cons. 3D Neighbor
- fanc
- fbnc
- M.J. Potential
- Access-Dependent Volume
- laar

### Requirements for Calculating Sequence-Based Features:

- **Three sequence-based features (Blosum62, PSSM, Shannon's entropy)** (.ARFF format):
    - The three sequence/MSA based features can be retrieved using the `PatMut.py` pipeline that is explained with a detailed description [here](https://github.com/NataliaSirera/patmut).

    - An example to execute the PatMut script is given below:
     ```sh
    python bin/PatMut.py demo/Q9Y375.config
    ```
    - The output of this script, ARFF file, includes a list of features for the variants of the protein. An example output can be found [here](/demo/Q9Y375.arff).
 
- Neco feature
    - `feature_neco.csv` available in `QAFI/data` folder
    - Please refer to our QAFI paper to follow how this dataframe was curated
    - Credits for the calculation of this feature: [Natàlia Padilla Sirera](https://github.com/NataliaSirera)

     
### Requirements for Calculating Structure-Based Features:

- **Colasi feature**:
    - **AlphaFold Structure** for the protein of interest (.mmCIF file format):
        - The structure can be downloaded from [AlphaFold](https://alphafold.ebi.ac.uk)
        - Refer to:
            - Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.

            - Varadi, M. et al. (2024). AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences. *Nucleic acids research*, 52(D1), D368-D375.


    - **Atomic Interactions within Protein Structures** (.json file format):
        - Follow the instructions provided at [Arpeggio GitHub](https://github.com/PDBeurope/arpeggio) to generate the necessary JSON file containing all atomic contacts.
        - Refer to: Jubb HC et al. (2017) Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. *J Mol Biol* 429:365–371.
              
    - Maximum number of contacts file
        - `MAX_COUNTS.csv` available in `QAFI/data` folder
        - The maximum number of interatomic contacts observed for the native residue’s type in a well-curated dataset of 593 experimental protein structures, list of structures selected from PISCES. For all pdb structures, atomic contacts are obtained, counted and maximum number of contact for each residue is stored in the "maximum_counts.csv" file.
        - For the source of selected experimental protein structures refer to: Wang G, Dunbrack RL (2003) PISCES: A protein sequence culling server. Bioinformatics 19:1589–1591

- **MJ potential feature**
    - `MJ_POTENTIAL_TABLE3.csv` available in `QAFI/data` folder
    - To calculate the difference in contact energies between the sums of native-neighbour and mutant-neighbour interactions, we used the upper triangle of the 20x20 table provided by Miyazawa and Jernigan (1996).
    - Refer to: Miyazawa S, Jernigan RL (1996) Residue-Residue Potentials with a Favorable Contact Pair Term and an Unfavorable High Packing Density Term, for Simulation and Threading. J Mol Biol 256:623–644.
      
- **Laar feature**
    - `feature_laar.csv` available in `QAFI/data` folder
    - Please refer to our QAFI paper to follow how this dataframe was curated
    - Credits for the calculation of this feature: [Natàlia Padilla Sirera](https://github.com/NataliaSirera)
  
---
# **Predictions with QAFI for a Protein of Interest**

The `Predictions_with_QAFI.ipynb` notebook includes a pipeline for obtaining predictions for a protein of interest, provided all its features are obtained.

QAFI predictions for 3460 proteins can be found in `output/QAFI_predictions/all` folder


**Note:** Please be aware that depending on the versions of BLAST, MUSCLE, and certain Python libraries used, the results generated by this script may exhibit slight variations.

---

## Reference

In case of using the QAFI predictor in your work, please use the following citation:
Selen Ozkan, Natàlia Padilla, Xavier de la Cruz et al. QAFI: A Novel Method for Quantitative Estimation of Missense Variant Impact Using Protein-Specific Predictors and Ensemble Learning, 07 May 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-4348948/v1]

---

**Disclaimer**: The predictions of all 30 PSPs and QAFI method are uniquely intended for research purposes. The authors are not responsible for its use or misuse. The data provided are not intended as advice of any kind. The authors have worked with care in the development of this dataset, but assume no liability or responsibility for any error, weakness, incompleteness, or temporariness of the data provided.
