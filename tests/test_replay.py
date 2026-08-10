import numpy as np

from sal_td3.replay import RankBasedNStepReplay


def test_three_step_return_and_terminal_flush():
    replay = RankBasedNStepReplay(capacity=20, n_step=3, gamma=0.5)
    for index in range(4):
        replay.add([index], [0.1, 0.2], 1.0, [index + 1], index == 3)
    assert len(replay) == 4
    rewards = sorted(round(item.reward, 3) for item in replay._storage)
    assert rewards == [1.0, 1.5, 1.75, 1.75]
    batch = replay.sample(4, beta=0.4)
    assert batch["states"].shape == (4, 1)
    assert np.all(batch["weights"] <= 1.0)
    replay.update_priorities(batch["indices"], np.arange(4, dtype=float))
