<div style="text-align: center; background-color: #FFFFFF;">
<p align="center">
  <img src="https://assets.zan.chat/sitev2/public/logo.png" alt="ZanChat AI logo" width="480"/>
</p>
</div>

<p align="center">
  <a href="https://zan.chat/fast-bagel">
    <img
      src="https://img.shields.io/badge/FastBAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="ZanChat AI Fast-BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/FastBAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="Fast-BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/zanchat-ai/fast-bagel">
    <img 
        src="https://img.shields.io/badge/FastBAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" 
        alt="Fast-BAGEL on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="Original BAGEL Demo"
    />
  </a>
  <a href="mailto:info@zan.chat">
    <img
      src="https://img.shields.io/badge/FastBAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="Fast-BAGEL Email"
    />
  </a>
</p>

# Fast-BAGEL: Fast and Detail-Enhanced Image Editing of Bagel

Fast-BAGEL is an improved version of LLM Image Editing/Generation based on Bagel. It reduces the inference step numbers, and improve the area details.

![Inference speed improvement between Fast-BAGEL vs. BAGEL](test_images/f4-steps.png)

We adopt Local-Gaussian Noise Coupling (LGNC) to retrain BAGEL with very few data. This method can be easily applied to other scenarios like frame-based video generation and 3D generation. It is very easy to replace flow-matching with our version.

```python
sigma = 0.9
epsilon = torch.randn_like(packed_latent_clean)
align_epsilon = epsilon * latent_std + latent_mean
coupled_noise = packed_latent_clean + sigma * align_epsilon
```

This simple code converts the flow matching noise to the Local-Gaussian noise used in Fast-BAGEL.

Fast-Bagel get similar or comparable results with only 50 steps compared with BAGEL 100 steps. You can refer to BAGEL demo for results and run our codes for speedup.

To run the demo, just run

```shell
CUDA_VISIBLE_DEVICES=0 python inference_ds.py --model_dir=/some/path/fast-bagel --num_timesteps=10,20,30 --time_shift=4
```

Though we released all the codes, please refer BAGEL for latest train/finetune details, as we may use some old codes. We recommend you to adopt our works direct to your current project, as BAGEL itself is not very suitable for direct production because of the code complexity and train dataset.

## ✍️ Citation

```bibtex
@article{liu2025-fast-bagel,
  title   = {LGCC: Enhancing Flow Matching Based Text-Guided Image Editing with Local Gaussian Coupling and Context Consistency},
  author  = {Fangbing Liu, Pengfei Duan, Wen Li, Yi He},
  journal = {arXiv preprint arXiv:2505.todo},
  year    = {2025}
}
```


# old contents of original BAGEL

**Original README from ByteDance Bagel**
see [OLD_README.md](OLD_README.md)