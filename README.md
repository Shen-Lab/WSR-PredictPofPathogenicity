# WeaklySupervisedRegressor


## The *Data* folder contains all raw data(including features and labels)
## The different feature sets of curated variants from ClinVar are stored in folder with naming pattern like:  
**features**\_last-interpreted-date(**06-29-2017**)\_**geq**(greater and equal to)**2**/**3**(stars)\_(**wholeFeature**)  

* Folders end with _wholeFeature_ contain 71 features with names listed below:  

Effect, Structure_and_dynamics, Secondary_structure, Stability_and_conformational_flexibility, Conformational_flexibility, Special_structural_signatures, Signal_peptide, Transmembrane_protein, Functional_residue, Macromolecular_binding, Protein_binding, Disordered_interface, Ordered_interface, Metal_binding, PTM_site, Intrinsic_disorder, B-factor, Relative_solvent_accessibility, Helix, Strand, Loop, N-terminal_signal, Signal_helix, C-terminal_signal, Signal_cleavage, Cytoplasmic_loop, Transmembrane_region, Non_cytoplasmic_loop, Coiled_coil, Catalytic_site, Calmodulin_binding, DNA_binding, RNA_binding, PPI_residue, PPI_hotspot, MoRF, Allosteric_site, Cadmium_binding, Calcium_binding, Cobalt_binding, Copper_binding, Iron_binding, Magnesium_binding, Manganese_binding, Nickel_binding, Potassium_binding, Sodium_binding, Zinc_binding, Acetylation, ADP-ribosylation, Amidation, C-linked_glycosylation, Carboxylation, Disulfide_linkage, Farnesylation, Geranylgeranylation, GPI_anchor_amidation, Hydroxylation, Methylation, Myristoylation, N-terminal_acetylation, N-linked_glycosylation, O-linked_glycosylation, Palmitoylation, Phosphorylation, Proteolytic_cleavage, Pyrrolidone_carboxylic_acid, Sulfation, SUMOylation, Ubiquitylation, Stability

* Other folders contain 9 features (a subset of these 71 features. See section 2.3 in the paper) with names listed below:  

Relative_solvent_accessibility, Allosteric_site, Catalytic_site, Secondary_structure, Stability_and_conformational_flexibility, Special_structural_signatures, Macromolecular_binding, Metal_binding, PTM_site

## The different label sets of curated variants are stored in folder with naming pattern like:  
**labels**\_last-interpreted-date(**06-29-2017**)\_**geq**(greater and equal to)**2**/**3**(stars)  

* Five classes are represented with number one to five:  

Benign - 1  
Likely Benign - 2  
Uncertain - 3  
Likely Pathogenic - 4  
Pathogenic - 5  

## Features and labels of CAGI test set are in CAGI_test folder  
* Files with _whole_ contain 71 features
* Files without _whole_ contain 9 features

#### Files
* Each file contains data for one gene. 
* Files ending with _clean_ only contains feauture matrix
* Files without _clean_ additionally contains variant names and header information

## Revision data set
The new data set collected for revision process is in folder /data_revision
#### New genes
We extended the scale of gene candidates to general genes availble in ClinVar dataset. The statistics for new gene set is shown below. For more details, check the file **SI\_new\_genes\_stat.xlsx**.

* Benign - 460(9.52%)
* Likely Benign - 694(14.36%)
* Uncertain - 2806(58.05%)
* Likely Pathogenic - 282(5.83%)
* Pathogenic - 592(12.25%)
* total - 4834

In order to test generalization of the model on genes sharing no connection with BRCA1/2, the 9 old property features for this new gene set are used to train the model and compared with the performance of model trained on same feature set of old genes. Experiment shows comparable performance indicating good generalization of our model.    

#### New features
Besides our old property features from Mutpred2, we ensambled more from dbNSFP database, including 6 pathogenicity scores and 8 conservation scores. 
* 6 pathogenicity scores: SIFT_score, Polyphen2_HDIV_score, Polyphen2_HVAR_score, PROVEAN_score, REVEL_score, PrimateAI_score
* 8 conservation scores: GERP++\_RS, phyloP100way_vertebrate, phyloP30way_mammalian, phyloP17way_primate, phastCons100way_vertebrate, phastCons30way_mammalian, phastCons17way_primate, bStatistic

To test whether we can have gain of performance if more features are included, this new 14 features is appended to old 9 features on old gene set and fed into the model. The result shows improvement over model trained on only 9 old features.  
