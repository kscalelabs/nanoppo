<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/ksim/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/min-consistency-models/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/min-consistency-models/actions/workflows/test.yml)

</div>

Minimial implementation of PPO for learning a robotics policy in Jax.

https://github.com/user-attachments/assets/ba305d28-2358-4fc3-8e4f-cdd103decfeb

# NanoPPO

This is a minimalist implementation of training and inferencing a PPO policy for making humanoid robots walk.

## How do I run this?

0. Install dependencies: `pip install --pre -r requirements.txt`
1. Start training a model: `python -m train`
2. Run inference on the trained model: `python -m play`
video will be saved to video.mp4

## Contributing

Before committing changes, run the following formatting commands:

```bash
make format
make static-checks
```
