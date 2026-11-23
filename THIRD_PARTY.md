# Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP), MIT license: the pretrained
  encoder, tokenizer, and checkpoint loader are installed as an external
  dependency; their upstream license remains applicable.
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412), Foret et al.,
  ICLR 2021: SBA uses the standard first-order SAM update with AdamW.
- [CAMS](https://github.com/ybyangjing/CAMS): the compositional split conventions,
  image preprocessing and float32 seen/unseen bias-curve protocol are followed.
- [Troika](https://github.com/bighuang624/Troika): optional open-world
  feasibility scores and the multiplicative filtering convention. Feasibility
  artifacts are obtained separately and are not bundled.

These references identify reused infrastructure/protocols, not claims that SBA
reimplements those complete methods. Dataset owners retain their dataset terms.
