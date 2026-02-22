# MedVerse: Efficient and Reliable Medical Reasoning via DAG-Structured Parallel Execution

<div align="center">

Bridging the gap between Medical Reasoning and Parallel Inference.

</div>

## 🔥 News

- **[02/07/2026]** MedVerse paper was released on [arXiv](https://arxiv.org/abs/2602.07529)!

<p align="center">
<img>
</p>

</h1>
</div>

<div align="center">
[<a href="https://arxiv.org/abs/2506.09991">📄 Paper</a>] | [<a href="https://multiverse4fm.github.io/">🌐 Website</a>] | [<a href="https://huggingface.co/Multiverse4FM">🤗 Huggingface</a>] | [<a href="https://x.com/Multiverse4FM">🐦 Twitter</a>]
</div>
<br>

## ⚡ TL;DR

Multiverse is a generative modeling framework that natively supports parallel generation for efficient test-time scaling. We provide an end-to-end ecosystem for building and deploying Multiverse models in real-world applications.

## 🎬 Demo

We showcase a Multiverse model solving a math reasoning problem, demonstrating its parallel generation capabilities.

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

## 📝 Todo List

- [ ] Add evaluation code based on lighteval
- [ ] Support context parallelism
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
