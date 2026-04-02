# Unsupervised Morphological and Syntactic Structure Induction via VAE

Unsupervised learning of linguistic morphology and syntax using a
sequence-to-sequence Variational Autoencoder (VAE).

---

## Project Overview

This project implements a VAE that learns, without any supervision,
to encode morphological and syntactic regularities into a structured
latent space. The latent vector `z` is partitioned into a morphological
subspace `z_morpho` and a syntactic subspace `z_syntax`. Post-training
analysis measures how strongly each subspace correlates with gold
linguistic annotations from Universal Dependencies.

---

## Setup First Posibility

### 1. Clone and create the environment
```bash
conda env create -f environment.yml
conda activate vae_morphosyntax
```

### 2. Download the data

**Universal Dependencies (English EWT and French GSD)**
```bash
git clone https://github.com/UniversalDependencies/UD_English-EWT.git data/raw/ud_english
git clone https://github.com/UniversalDependencies/UD_French-GSD.git data/raw/ud_french
```

**Morpho Challenge 2010**
```
Download from: http://morpho.aalto.fi/events/morphochallenge2010/
Place files in:  data/raw/morpho_challenge/
```

**UniMorph (English)**
```bash
git clone https://github.com/unimorph/eng.git data/raw/wiktionary
```

**FastText embeddings (optional)**
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P data/embeddings/
gunzip data/embeddings/cc.en.300.bin.gz
```

### 3. Build C++ and CUDA extensions
```bash
python setup.py build_ext --inplace
```

### 4. Run the full pipeline

Open `main.ipynb` and run all cells in order, or run each section
independently using the auxiliary notebooks in `notebooks/`.

## Running Tests
```bash
pytest tests/ -v --cov=src
```








## Setup Second Posibility

### Step 1 — Create the conda environment
```bash
conda env create -f environment.yml
conda activate image_mining
```

### Step 2 — Download the data
```bash
# Universal Dependencies
git clone https://github.com/UniversalDependencies/UD_English-EWT.git data/raw/ud_english
git clone https://github.com/UniversalDependencies/UD_French-GSD.git data/raw/ud_french

# UniMorph
git clone https://github.com/unimorph/eng.git data/raw/wiktionary

# FastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P data/embeddings/
gunzip data/embeddings/cc.en.300.bin.gz

# Morpho Challenge → manual download at:
# http://morpho.aalto.fi/events/morphochallenge2010/
# Place files in: data/raw/morpho_challenge/
```

### Step 3 — Compile C++/CUDA extensions
```bash
cd cpp_extensions/
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ../..
```

### Step 4 — Install the Python package
```bash
pip install -e . --no-build-isolation
```

### Step 5 — Download NLP resources
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('cmudict')"
```

### Step 6 — Launch the main notebook
```bash
jupyter notebook main.ipynb
```
Then run all cells in order via `Cell → Run All`.

### Step 7 — Run analysis notebooks (optional, in this order)
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_model_architecture.ipynb
jupyter notebook notebooks/03_latent_space_analysis.ipynb
jupyter notebook notebooks/04_linguistic_evaluation.ipynb
```

### Step 8 — Run the tests
```bash
pytest tests/ -v --cov=src
```

---

## References

- Bowman et al. (2016). Generating Sentences from a Continuous Space.
- Higgins et al. (2017). beta-VAE: Learning Basic Visual Concepts.
- Fu et al. (2019). Cyclical Annealing Schedule for VAEs.
- Eastwood & Williams (2018). A Framework for Disentangled Representations.
- Chen et al. (2018). Isolating Sources of Disentanglement in VAEs.
- Nivre et al. (2020). Universal Dependencies v2.
- Kurimo et al. (2010). Morpho Challenge 2010.
