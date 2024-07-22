# NanoPPO

This is a minimalist implementation of training and inferencing a PPO policy for making humanoid robots walk.

## How do I run this?

0. Install dependencies: `pip install -r requirements.txt`
1. Start training a model: `python train.py`
2. Run inference on the trained model: `python infer.py`

## Contributing

Before committing changes, run the following formatting commands:

```bash
black train.py infer.py
ruff format train.py infer.py
```
