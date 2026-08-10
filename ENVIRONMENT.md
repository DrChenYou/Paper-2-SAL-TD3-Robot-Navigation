# Environment and data contract

## Paper setup

The simulation study used ROS and Gazebo on Ubuntu 20.04 LTS with a TurtleBot3 Waffle Pi and a 360-degree LiDAR producing 360 range measurements. The principal map was 10 m by 10 m. Dynamic obstacle speeds were varied in the 0.2-0.6 m/s range, with direction updates and a partially pursuit-biased motion mode. Training used 1,500 episodes with at most 500 steps per episode.

The reported workstation contained an AMD Ryzen 5 5600X, an NVIDIA RTX 3060 with 12 GB VRAM, and 32 GB RAM. Real-world validation used an AgileX BUNKER AGV with an RS-Helios-16P LiDAR in an 8 m by 8 m environment.

## Adapter contract

A simulator or robot adapter should return:

- `lidar`: 360 finite ranges in metres, clipped to the configured sensor maximum;
- `goal_pose`: goal direction/distance and robot pose features in a documented order;
- `previous_distance` and `current_distance`: Euclidean goal distances in metres;
- `heading_error`: signed angular error in radians within `[-pi, pi]`;
- `minimum_obstacle_distance`: closest valid range in metres;
- `goal_reached`, `collision`, and `truncated`: Boolean termination fields.

Maintain the latest observations in a fixed-length history. A replay state is therefore shaped `[sequence_length, state_dim]`; the actor and critics also accept a single state and add a length-one sequence automatically.

The two action components are normalized to `[0, 1]` by the actor. The adapter must document how they map to the platform's permitted linear and angular velocities. Apply physical safety limits outside the learning agent.

## Local layout

```text
data/
|-- worlds/       # Gazebo world descriptions
|-- rosbags/      # Optional recorded sensor streams
|-- checkpoints/  # Model and optimizer states
`-- evaluations/  # Per-trial metrics and trajectories
```

These paths are ignored by Git because robot logs, environments, and checkpoints can be large or access-controlled.

## Evaluation protocol

For every scenario, freeze the trained policy, disable exploration noise, and run at least 100 independent trials. Preserve the seed and per-trial outputs. A run record should include success/collision status, path length, duration, minimum clearance, map identifier, start/goal pose, obstacle seed, checkpoint hash, and software versions.
