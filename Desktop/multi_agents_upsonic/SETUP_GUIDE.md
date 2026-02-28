# Complete Setup Guide - AI Learning Journey Portfolio

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone or Download Project
```bash
cd c:\Users\Kullanıcı\Desktop\multi_agents_upsonic
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
"
```

### Step 5: Launch Jupyter
```bash
jupyter notebook
```

Open your browser to `http://localhost:8888` and navigate to the notebooks.

---

## 📋 System Requirements

### Hardware Minimum
- **CPU:** Intel i5 or equivalent (2+ cores)
- **RAM:** 4GB (8GB recommended)
- **Storage:** 2GB free space
- **GPU:** Optional (significantly speeds up deep learning)

### Software Prerequisites
- **Python:** 3.8 or higher
  ```bash
  python --version
  ```
- **pip:** Package installer (comes with Python 3.8+)
  ```bash
  pip --version
  ```

### Operating Systems Supported
- ✅ Windows 10/11
- ✅ macOS (Intel or Apple Silicon)
- ✅ Linux (Ubuntu, Debian, etc.)

---

## 🔧 Detailed Installation Steps

### Windows Installation

#### 1. Install Python
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Choose "Install Now"

#### 2. Verify Installation
```cmd
python --version
pip --version
```

#### 3. Create Project Directory
```cmd
mkdir multi_agents_upsonic
cd multi_agents_upsonic
```

#### 4. Virtual Environment Setup
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see: `(venv) C:\Users\...>`

#### 5. Upgrade pip
```cmd
python -m pip install --upgrade pip
```

#### 6. Install Dependencies
```cmd
pip install -r requirements.txt
```

#### 7. Verify Installation
```cmd
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import pandas; print(pandas.__version__)"
```

### macOS Installation

#### 1. Install Homebrew (Package Manager)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Python via Homebrew
```bash
brew install python
```

#### 3. Rest of Steps Same as Linux Below
```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian) Installation

#### 1. Update System
```bash
sudo apt-get update
sudo apt-get upgrade
```

#### 2. Install Python Development Tools
```bash
sudo apt-get install python3 python3-pip python3-venv build-essential
```

#### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Install Additional System Library (for PyPDF2)
```bash
sudo apt-get install libpdf-dev
```

---

## 🐍 Virtual Environment Guide

### Why Use Virtual Environments?

Virtual environments isolate project dependencies to avoid conflicts:

```
System Python
├─ Project A Environment (Python 3.8 + specific packages)
├─ Project B Environment (Python 3.9 + different packages)
└─ Project C Environment (Python 3.10 + other packages)
```

### Creating Virtual Environment

**Windows:**
```cmd
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

### Activating Virtual Environment

**Windows:**
```cmd
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Deactivating Virtual Environment
```bash
deactivate
```

### Checking Active Environment
```bash
which python  # macOS/Linux shows venv location
where python  # Windows shows venv location
```

---

## 📦 Dependency Management

### Understanding requirements.txt

```txt
# Format: package_name>=minimum_version

# Core Data Science
pandas>=1.3.0      # Data manipulation
numpy>=1.21.0      # Numerical computing
scikit-learn>=1.0.0 # Machine learning

# NLP specific
nltk>=3.6.0        # Tokenization, stemming
textblob>=0.17.0   # Sentiment analysis

# Deep Learning
tensorflow>=2.8.0  # Neural networks
```

### Installing Specific Package

```bash
# Install single package
pip install tensorflow

# Install specific version
pip install tensorflow==2.10.0

# Install with extras
pip install tensorflow[and-cuda]
```

### Updating Packages

```bash
# Update single package
pip install --upgrade pandas

# Update all packages
pip list --outdated
pip install --upgrade -r requirements.txt
```

### Creating Your Own Requirements File

```bash
# After installing packages
pip freeze > requirements.txt

# Only for this project's packages (removes system packages)
pip freeze > requirements.txt
```

---

## 📓 Jupyter Notebook Setup

### Launch Jupyter

```bash
jupyter notebook
```

This opens browser at `http://localhost:8888`

### Alternative: JupyterLab (Enhanced UI)

```bash
jupyter lab
```

Opens at `http://localhost:8888/lab`

### Navigation

1. **Files Panel (Left):** Browse project directory
2. **Notebook:** Click to open `.ipynb` files
3. **Terminal (Optional):** Tools → Open Terminal

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run current cell |
| `Shift+Enter` | Run cell and move to next |
| `Alt+Enter` | Run cell and insert new below |
| `Esc` | Exit edit mode |
| `DD` | Delete cell |
| `M` | Change to Markdown |
| `Y` | Change to Code |

---

## 🧬 Data Files Setup

### Required Datasets

Ensure these files are in your project directory:

```
multi_agents_upsonic/
├── data/
│   ├── yelp.csv              # Yelp reviews dataset
│   ├── spam.csv              # Spam classification data
│   ├── sgk_drugs_unique.json  # Pharmaceutical data
│   └── [other data files]
│
├── notebooks/
│   ├── NLP-1.ipynb
│   ├── NLP-Fetching.ipynb
│   └── NLPwithDL.ipynb
```

### Checking Data Files

```python
import os
import pandas as pd

# Verify files exist
print("Files in directory:")
for file in os.listdir('data/'):
    print(f"  - {file}")

# Load datasets
df = pd.read_csv('data/yelp.csv')
print(f"Yelp data shape: {df.shape}")
```

---

## 🧠 NLTK Data Download

NLTK requires separate data downloads beyond package installation:

### Automatic Download (Recommended)

```python
import nltk
import ssl

# Bypass SSL certificate issues if needed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_unverified_context = _create_unverified_https_context

# Download all necessary data
nltk.download('punkt')                      # Tokenizers
nltk.download('punkt_tab')                  # Updated tokenizer
nltk.download('stopwords')                  # English stopwords
nltk.download('wordnet')                    # Lemmatization database
nltk.download('averaged_perceptron_tagger') # POS tagger
```

### Manual Download via NLTK GUI

```python
import nltk
nltk.download()  # Opens GUI for selection
```

### Verify Downloads

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Test tokenization
text = "Hello world!"
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")  # Should work without errors

# Test stopwords
stops = stopwords.words('english')
print(f"Downloaded {len(stops)} stopwords")

# Test lemmatizer
lemma = WordNetLemmatizer()
print(f"Lemmatized 'running': {lemma.lemmatize('running', 'v')}")
```

---

## ⚠️ Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow --upgrade
```

**Alternative (if GPU needed):**
```bash
pip install tensorflow[and-cuda]
```

### Issue: NLTK Data Not Found

**Solution:**
```python
import nltk
nltk.data.path.append('/path/to/nltk_data')
```

Or set environment variable:
```bash
# Windows
set NLTK_DATA=C:\nltk_data

# macOS/Linux
export NLTK_DATA=/home/user/nltk_data
```

### Issue: PDF Reading Fails

**Solution:**
```bash
# Reinstall with fresh installation
pip uninstall PyPDF2
pip install PyPDF2 --upgrade
```

### Issue: Memory Error with Large Datasets

**Solution:**
```python
# Process in chunks
import pandas as pd

for chunk in pd.read_csv('data.csv', chunksize=1000):
    process_chunk(chunk)
```

### Issue: Jupyter Notebook Kernel Dies

**Solution:**
```bash
# Restart kernel
jupyter notebook --NotebookApp.custom_display_url='http://localhost:8888'

# Or clear Jupyter cache
jupyter kernelspec list
jupyter kernelspec remove python3
python -m ipykernel install --user
```

### Issue: GPU Not Being Used by TensorFlow

**Verify GPU:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If empty, install CUDA:
```bash
# Requires NVIDIA GPU and CUDA installation
pip install tensorflow[and-cuda]
```

---

## 🔍 Verification Checklist

After setup, verify everything works:

```python
# verification_script.py
import sys

print("=" * 50)
print("VERIFICATION CHECKLIST")
print("=" * 50)

# Python Version
print(f"✓ Python: {sys.version}")

# Core Libraries
try:
    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")
except ImportError:
    print("✗ pandas not installed")

try:
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
except ImportError:
    print("✗ numpy not installed")

try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn not installed")

# NLP Libraries
try:
    import nltk
    print(f"✓ NLTK installed")
except ImportError:
    print("✗ NLTK not installed")

try:
    from textblob import TextBlob
    print(f"✓ TextBlob installed")
except ImportError:
    print("✗ TextBlob not installed")

# Deep Learning
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not installed")

# GPU Check
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU: {len(gpus)} device(s) found")
    else:
        print("⚠ GPU: No NVIDIA GPU found (CPU only)")
except:
    print("⚠ GPU check failed")

print("\n" + "=" * 50)
print("Setup verification complete!")
print("=" * 50)
```

Run verification:
```bash
python verification_script.py
```

---

## 🎯 Next Steps After Setup

1. **Run NLP-1.ipynb** - Start with fundamentals
2. **Run NLP-Fetching.ipynb** - Learn data acquisition
3. **Run NLPwithDL.ipynb** - Implement deep learning
4. **Experiment** - Modify hyperparameters, try new data
5. **Extend** - Add new features or models

---

## 💡 Pro Tips

### Speeding Up Training
```python
# Use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Reduce data for testing
df = df.sample(n=1000)

# Use smaller model
Dense(64, activation='relu')  # Instead of 128
```

### Managing Disk Space
```bash
# Remove unused packages
pip uninstall -y tensorflow-cpu tensorflow

# Clean pip cache
pip cache purge

# Remove virtual environment when done
rm -rf venv/  # macOS/Linux
rmdir /s venv  # Windows
```

### Collaborative Development
```bash
# Save exact versions (for others to replicate)
pip freeze > requirements-exact.txt

# Share with team
git add requirements.txt
git commit -m "Update dependencies"
git push
```

---

## 📚 Additional Resources

- [Python Official Docs](https://docs.python.org/3/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [NLTK Book](https://www.nltk.org/book/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Jupyter Documentation](https://jupyter.readthedocs.io/)

---

**Happy Learning! 🚀**

If you encounter any issues, refer to the Troubleshooting section or check official package documentation.
