# Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models
[Eric Zhang]()<sup>&dagger;</sup>, [Kai Wang](https://wangk.ai/)<sup>&dagger;</sup>, [Xingqian Xu](https://ifp-uiuc.github.io/), [Zhangyang Wang](), [Humphrey Shi](https://www.humphreyshi.com/home)

<sup>&dagger;</sup> Equal Contribution

[[`arXiv`]()] [[`pdf`]()]

This repo contains the code for our paper **Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models**.

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
