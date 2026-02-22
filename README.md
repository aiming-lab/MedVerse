# MedVerse: Efficient and Reliable Medical Reasoning via DAG-Structured Parallel Execution

<div align="center">

Bridging the gap between Medical Reasoning and Parallel Inference.

</div>

## 🔥 News

- **[02/07/2026]** MedVerse paper was released on [arXiv](https://arxiv.org/abs/2602.07529)!

## 📖 Overview

MedVerse is a framework that enables LLM agents to learn high-level, reusable behavioral patterns from past experiences. While traditional memory-based methods store redundant and noisy raw trajectories, SKILLRL abstracts these into a hierarchical skill library.

## 🤖 Key Features

- **Experience-based Skill Distillation**: Transforms successful trajectories into strategic patterns and failed ones into concise lessons from failure.

- **Hierarchical SKILLBANK**: Organizes knowledge into General Skills for universal strategic guidance and Task-Specific Skills for category-level heuristics.

- **Recursive Skill Evolution**: A dynamic mechanism where the skill library co-evolves with the agent's policy during RL by analyzing validation failures.

- **Context Efficiency**: Achieves 10-20% token compression compared to raw trajectory storage while enhancing reasoning utility. 

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/aiming-lab/MedVerse.git
cd MedVerse

pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .

pip install openai
```

## 🏛️ Repository Structure

This repository provides a complete ecosystem for building and deploying Multiverse models. Our structure is organized as follows:

**🗂️ `data`** → **📈 `train`** → **🚀 `inference`**

```
Multiverse
├── data/
│   └── src
|   └── run
|   └── README.md
│
├── train
│   └── README.md
│
├── inference/
│   └── engine
|   └── README.md
│
└── README.md
```

- **`data/`**: Contains the **Multiverse Curator** toolkit for dataset preparation. Use it to generate your own **Multiverse-1K** dataset for training.

- **`training/`**: Implements the **Multiverse Attention** algorithm for the efficient training of Multiverse models. We also includes the code for AR baselines

- **`inference/`**: Features the **Multiverse Engine** implementation, a high-performance inference server optimized for Multiverse models.

For detailed documentation and usage instructions, please refer to the README.md files in each directory.

## 📚 References

Thank you for your interest in Multiverse Engine! We hope this tool will be helpful for your research and development. If you find it useful, please consider citing our work. Happy coding! 🚀

```bibtex
@misc{yang2025multiverselanguagemodelssecretly,
      title={Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation}, 
      author={Xinyu Yang and Yuwei An and Hongyi Liu and Tianqi Chen and Beidi Chen},
      year={2025},
      eprint={2506.09991},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.09991}, 
}
```
