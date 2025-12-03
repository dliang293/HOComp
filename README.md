# HOComp: Interaction-Aware Human-Object Composition  
### *NeurIPS 2025*

This is the official repository for our paper:

📄 **HOComp: Interaction-Aware Human-Object Composition**  
📚 *Preprint available on arXiv*

---

## 📝 Abstract
**HOComp** is a novel framework for harmonizing foreground objects into human-centric backgrounds.  
By leveraging a **Flux.1 Kontext** base model and a novel **Sequence Concatenation** strategy, the method achieves precise control over **human–object interactions** with high fidelity.

---

## 🛠️ Custom Inference

To generate a specific interaction, provide background / foreground images, the interaction prompt, and the foreground bounding box:

```bash
python run_inference.py \
  --prompt "A young man holding a vintage camera" \
  --bg_path "examples/background.jpg" \
  --fg_path "examples/camera.png" \
  --box "[300 300 700 700]" 
```


## 📌 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{liang2025hocomp,
  title={HOComp: Interaction-Aware Human-Object Composition},
  author={Dong Liang and Jinyuan Jia and Yuhao Liu and Rynson W. H. Lau},
  journal={arXiv preprint arXiv:2507.16813},
  year={2025}
}