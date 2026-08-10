# Model card: SAL-TD3 navigation policy

## Model summary

SAL-TD3 is a continuous-control policy for dynamic mobile-robot navigation. It combines a multimodal LiDAR/goal encoder, an LSTM temporal model, four-head self-attention, twin recurrent critics, delayed policy updates, rank-based prioritized replay, and three-step returns.

## Intended use

- Research on robot navigation and sequential reinforcement learning.
- Controlled simulation experiments and supervised real-world validation.
- Ablation studies of temporal modelling, attention, replay prioritization, and multi-step targets.

## Out-of-scope use

Do not deploy an unchecked policy around people, vehicles, public infrastructure, or safety-critical equipment. This implementation is not a certified collision-avoidance or functional-safety system.

## Inputs and outputs

The learning networks consume fixed-length state histories. A typical state combines encoded 360-degree LiDAR data with goal and pose features. The actor returns two normalized values for platform-specific linear and angular velocity mapping.

## Limitations

- Performance depends on sensor calibration, observation history, robot dynamics, action scaling, simulator fidelity, and obstacle behaviour.
- A success rate measured in one map does not establish safety in a new environment.
- The paper's real-world platform and Gazebo assets are not interchangeable with arbitrary robots without adaptation and new validation.
- Reflective surfaces, occlusion, dropped scans, localization error, and unmodelled human motion may degrade performance.

## Evaluation

Report success and collision rates alongside path length and runtime over independently seeded trials. Include results for unseen maps and dynamics. For physical trials, use conservative speed limits, an independent emergency stop, a controlled test area, and qualified supervision.
