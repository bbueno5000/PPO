# Proximal Policy Optimization (PPO)

PPO implementation for an OpenAI gym environment.

The template for the brain and environment were derived from [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents).

Notable changes include:
  * Ability to continuously display progress with non-stochastic policy during training
  * Works with OpenAI environments
  * Option to record episodes
  * State normalization for given number of frames
  * Frame skip
  * Faster reward discounting etc.

## Environment Setup

[environment_setup.md](docs\environment_setup.md) contains instructions for getting started.

## Run Program

```
python main.py
```

## Best Practices

[best_practices.md](docs\best_practices.md) contains guidelines for implementing PPO.

## Credits

* **Sven Niederberger** - [EmbersArc](https://github.com/EmbersArc) - *original author*
* **Benjamin Bueno** - [bbueno5000](https://github.com/bbueno5000) - *contributor*
