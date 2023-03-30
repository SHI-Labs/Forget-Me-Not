# Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models
[Eric Zhang]()<sup>&dagger;</sup>, [Kai Wang](https://wangk.ai/)<sup>&dagger;</sup>, [Xingqian Xu](https://ifp-uiuc.github.io/), [Zhangyang Wang](), [Humphrey Shi](https://www.humphreyshi.com/home)

<sup>&dagger;</sup> Equal Contribution

[[`arXiv`]()] [[`pdf`]()]

This repo contains the code for our paper **Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models**.

The significant advances in text-to-image generatve models have prompted global discussions on privacy, copyright, and safety, as numerous unauthorized personal IDs, content, artistic creations, and potentially harmful materials have been learned by these models and later utilized to generate and distribute uncontrolled content. 

To address this challenge, we propose **Forget-Me-Not**, an efficient and low-cost solution designed to safely remove specified IDs, objects, or styles from a well-configured text-to-image model in as little as 30 seconds, without impairing its ability to generate other content. Alongside our method, we introduce the **Memorization Score (M-Score)** and **ConceptBench** to measure the models’ capacity to generate general concepts, grouped into three primary categories: ID, object, and style. Using M-Score and ConceptBench, we demonstrate that Forget-Me-Not can effectively eliminate targeted concepts while maintaining the model’s performance on other concepts. Furthermore, Forget-Me-Not offers two practical extensions: a) removal of potentially harmful or NSFW content, and b) enhancement of model accuracy, inclusion and diversity through concept correction and disentanglement.
It can also be adapted as a lightweight model patch for Stable Diffusion, allowing for concept manipulation and convenient distribution. 

We hope our research and open-source here encourage future research in this critical area and promote the development of safe and inclusive generative models.

<img src="images/teaser.png" width="100%"/>

#### Features

- Forget-Me-Not is a plug-and-play, efficient and effective concept forgetting and correction method for large-scale text-to-image models.
- It provides an efficient way to forget specific concepts with as few as 35 optimization steps, which typically takes about 30 seconds.
- It can be easily adapted as lightweight patches for Stable Diffusion, allowing for multi-concept manipulation and convenient distribution.
- Novel attention re-steering loss demonstrates that pretrained models can be further finetuned solely with self-supervised signals, i.e. attention scores.

![Forget-Me-Not](images/attn-resteering.png)

## News

- **[March 30, 2023]**: Code is coming soon!

## Results

![Concept-Forgetting-Results](images/extra-results.png)

## Citation

If you found Forget-Me-Not useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex

```

## Acknowledgement

We thank the authors of [Diffusers](https://github.com/huggingface/diffusers) and [LoRA](https://github.com/cloneofsimo/lora) for releasing their helpful codebases.
