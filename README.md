# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation

This repository contains the implementation for the paper:
***LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation***

> [\[📄 arXiv:2505.12031\]](https://arxiv.org/abs/2505.12031)

---

## 📁 Directory Structure

```text
llm_base_atp/
├── README.md                    # Project documentation
├── Visual/                      # Proof search tree visualizations
├── DoBeVi/                      # Core implementation of DoBeVi
│   ├── requirements.txt         # Python dependencies
│   └── src/                     # Source code
│       ├── __init__.py          # Package initializer
│       ├── dojo/                # Lightweight LeanDojo wrapper for Lean 4 interaction
│       ├── search/              # Proof search algorithms and logic
│       ├── eval.py              # Entry point for evaluation
│       ├── config.py            # Configuration settings 
│       ├── utils.py             # Utility functions
│       └── .env.template        # Template for environment setup
```

---

## ⚙️ Setup Instructions

### 1. Create and Activate Conda Environment

```bash
conda create -n dobevi python=3.11 -y
conda activate dobevi
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Graphviz (Required for Visualization)

```bash
conda install -c conda-forge graphviz
```

### 4. Download the Policy Model

Please download the pretrained policy model from [Hugging Face](https://huggingface.co/NJUDeepEngine/llm_based_atp).

---

## 🧪 Evaluation

### Step 1: Prepare a Lean 4 Project

Ensure you have a Lean 4 repository that builds successfully using:

```bash
lake build
```

### Step 2: Configure Environment Variables

Copy and edit the template config file to match your local setup:

```bash
cd src/
cp .env.template .env
```

You’ll need to set values for:

* Benchmark project path
* Model path
* Tree search budget
* Output path for results

### Step 3: Run the Evaluation

```bash
python -m eval
```

---

## 🚧 TODO / Future Work

* [ ] Release code for synthetic data generation
* [ ] Add support for whole-proof methods and fine-tuning

---

## 🙋 Contributing & Issues
We welcome issues, feature requests, and feedback from the community. Please feel free to [open an issue](https://github.com/NJUDeepEngine/llm_based_atp/issues) if you encounter any problems or have suggestions!

---

## 🔗 Related Links

* [Hugging Face Model Page](https://huggingface.co/NJUDeepEngine/llm_based_atp)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{lai2025llm,
  title={LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation},
  author={Lai, Junyu and Zhang, Jiakun and Xu, Shuo and Chen, Taolue and Wang, Zihang and Yang, Yao and Zhang, Jiarui and Cao, Chun and Xu, Jingwei},
  journal={arXiv preprint arXiv:2505.12031},
  year={2025}
}
```