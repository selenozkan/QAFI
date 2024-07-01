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
# **Obtaining Features for a Protein of Interest**

All models developed in this study incorporate 14 features: 5 sequence-based and 9 structure-based.

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

Three sequence-based features (Blosum62, PSSM, Shannon's entropy) can be retrieved using the `PatMut.py` pipeline.
- *Note: Credits for the development of this pipeline go to Natàlia Padilla.*

Before running the `PatMut.py` pipeline, the following databases must be installed:

- **BLAST**: For protein sequence alignment. Download and install BLAST from [NCBI BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download).
- **MUSCLE**: For multiple sequence alignment. Download and install MUSCLE from [MUSCLE](https://drive5.com/muscle5/).
- **UniRef Database**: For retrieving protein sequences. Download the 2016 UniRef database from [UniProt](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2016_07/uniref/). We are updating to use the newest UniRef database but...

This pipeline is designed for the detailed analysis of protein variants. It begins by configuring settings specified in the `config.py` file and proceeds through several key stages. First, it builds a Multiple Sequence Alignment (MSA) using BLAST and MUSCLE if enabled. The script then parses variants from the MSA and retrieves all specified variants, categorizing them as pathological, neutral, or predicted mutations. Next, it calculates the desired sequence-based features by the user. Finally, `PatMut.py` writes analysis results to the specified output directory.

Below is an explanation of the key configuration parameters, followed by a link to a sample configuration file in this repository.

#### Configuration Parameters:

- **QUERY**
  - `uniprotID`: The UniProt identifier specific to the protein being analyzed. This parameter must be updated each time the analysis is conducted on a different protein.
  - `blastPath`: Directory path where the BLAST executable is located on the system.
  - `blastProtDB`: Location of the BLAST protein database file used for sequence similarity searches. Ensure this database (makeblastdb) is created using the UniRef 2016 database.

- **RESULTS**
  - `output`: Directory path where all results will be saved. 
  - `write`: Format(s) in which results will be saved. Supported formats include:
    - `MSA`: Generated in FASTA format (Q9Y375.msa).
    - `NeutresMSAID`: This format (Q9Y375.neutresMSA) provides a list of neutral mutations identified during the analysis, along with their associated sequence identifiers.
    - `ARFF`: Used for machine learning tasks, the ARFF file (Q9Y375.arff) includes attribute data derived from the user-configured features computed during the analysis. The tag attribute assigns a label (1 for pathological, 0 for neutral) to each variant based on its predicted impact.

- **MUTATION SET**
  - `pathological`: Source database for pathological mutations, sourced from the Humsavar database.
  - `neutral`: Method for identifying neutral mutations. This involves building a MSA (msaBuild option in the config file) to compare mutations across homologous sequences.
  - `predicting`: Mode for predicting mutations.

- **PROPERTIES**
  - Refer to the features calculated during the analysis and included in the output files.

- **DATABASES**
  - Paths to various supporting database files used in the analysis. The Humsavar database can be downloaded from [UniProt](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/humsavar.txt). The remaining databases are located in the metadata folder.

- **BUILD MSA**
  - Determines whether a MSA should be constructed for the protein of interest. Parameters such as `numIter`, `evalue`, `minSeqIden`, and `muscleAlg` control the specifics of the MSA process, including the number of iterations, alignment similarity thresholds (E-value and minimum sequence identity), and the path to the MUSCLE alignment tool.

- **Modelling Neutral Variants from MSA**
  - Specifies parameters `idmin` and `idmax`, defining the range of sequence identities used when modeling neutral variants from the MSA.
 
For instructions on configuring and running the script, refer to the sample configuration file available [here](demo/Q9Y375.config).

To execute the `PatMut.py` pipeline with the configuration file, use the following command: 
```
python3 PatMut.py /path/to/Q9Y375.config
```

### Requirements for Calculating Structure-Based Features:

- **AlphaFold Structure** for the protein of interest (.mmCIF file format):
    - The structure can be downloaded from [AlphaFold](https://alphafold.ebi.ac.uk)
    - Refer to:
        - Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.

        - Varadi, M. et al. (2024). AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences. *Nucleic acids research*, 52(D1), D368-D375.


- **Atomic Interactions within Protein Structures** (.json file format):
    - Follow the instructions provided at [Arpeggio GitHub](https://github.com/PDBeurope/arpeggio) to generate the necessary JSON file containing all atomic contacts.
    - Refer to:
        - Jubb HC et al. (2017) Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. *J Mol Biol* 429:365–371.

The notebook `Features.ipynb` includes a pipeline for obtaining all 14 features step-by-step for a given protein of interest.

---
# **Predictions with QAFI for a Protein of Interest**

The `Predictions_with_QAFI.ipynb` notebook includes a pipeline for obtaining predictions for a protein of interest, provided all its features are obtained.

---

## Reference

In case of using the QAFI predictor in your work, please use the following citation:
Ozkan, Selen, Natàlia Padilla, and Xavier de la Cruz. "QAFI: A Novel Method for Quantitative Estimation of Missense Variant Impact Using Protein-Specific Predictors and Ensemble Learning." (2024).

---

**Disclaimer**: The predictions of all 30 PSPs and QAFI method are uniquely intended for research purposes. The authors are not responsible for its use or misuse. The data provided are not intended as advice of any kind. The authors have worked with care in the development of this dataset, but assume no liability or responsibility for any error, weakness, incompleteness, or temporariness of the data provided.
