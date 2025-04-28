# CX4803 Project 🧬
## Classifying Antimicrobial Resistance Genes with Sequence-Derived Features
Team: Wanrong Qi, Fanshi Meng, Angela Huynh

Our project predicts antibiotic drug classes from protein sequences using a Random Forest classifier, impurity-based feature selection, and hyperparameter tuning.

### Files
- `preprocessing.ipynb` —  Parse CARD FASTA file, map sequences to drug classes using ARO index from metadata file, handle missing labels ("Unknown")
- `embeddings.py`— Reads unique, non-null protein sequences from CSV, encodes them using a pretrained transformer model (AutoTokenizer, AutoModel), mean-pools the hidden states to generate fixed-length embeddings, and base64-encodes the embeddings for storage in a new protein_embedding column
- `embeddings.csv.zip` — Compressed file of encoded embeddings
- `decoder.py` — File to decode embeddings
- `Random_forest_final.ipynb` — Loads the embedded dataset, addresses class imbalance by grouping rare classes into "other_antibiotics," engineers a wide range of sequence-based features (e.g., length, GC content, amino acid/nucleotide frequencies, purine/pyrimidine content), trains a baseline Random Forest model, performs feature selection using feature importance and RFECV, tunes hyperparameters with RandomizedSearchCV, trains a final optimized Random Forest model, and explores a soft-voting ensemble with Random Forest and Gradient Boosting classifiers.
- `card-data/`  — Contains data from Comprehensive Antibiotic Resistance Database (CARD)

### Workflow
`preprocessing.ipynb` ---> `embeddings.py` + `decoder.py` ---> `Random_forest_final.ipynb`

### Data
Data was retrieved from https://card.mcmaster.ca/download and is located in the `card-data/` directory

### Results
Output data and evalutation figures are stored in Output folder.
Discussion of results can be found in our report: https://docs.google.com/document/d/1d4C7N26u9hEGfQXaFGDVm6k4wd5Wdk3I_PnG1rc472M/edit?usp=sharing

### Contributions
- Data Formatting/Processing — Fanshi Meng, Angela Huynh
- Training the model + Hyperparameter Tuning — Wanrong Qi
- Model Evaluation + Report writing — All

