# EVA in Atari
 
Ephemeral Value Adjustment(EVA)

 ## Ephemeral Value Adjustment
[Fast deep reinforcement learning using online adjustments from the past](https://arxiv.org/pdf/1810.08163.pdf)

## Usage
- train.py

    Adapt to any combination of algorithms
    - --dueling Using dueling network
    - --n_step_return Set to numeric
    - --lambdas Set to 0 or 1 to not use non parametric at all(DQN)
    - --LRU Use LRU when storing at value buffer
    

## Requirement
- chainerrl
- chainer

