# NanoPPO

This is a minimalist implementation of training and inferencing a PPO policy for making humanoid robots walk.

## How do I run this?

0. Install dependencies: `pip install --pre -r requirements.txt`
1. Start training a model: `python -m mjx.train`
2. Run inference on the trained model: `python -m mjx.play`
video will be saved to video.mp4

## Contributing

Before committing changes, run the following formatting commands:

```bash
black *.py
ruff format *.py
isort .
```
