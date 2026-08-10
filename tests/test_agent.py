import numpy as np

from sal_td3.agent import SALTD3Agent


def test_agent_update_returns_priorities():
    rng = np.random.default_rng(7)
    agent = SALTD3Agent(6, hidden_dim=32, lstm_hidden_dim=16, policy_delay=1)
    shape = (4, 3, 6)
    batch = {
        "states": rng.normal(size=shape).astype("float32"),
        "actions": rng.uniform(size=(4, 2)).astype("float32"),
        "rewards": rng.normal(size=4).astype("float32"),
        "next_states": rng.normal(size=shape).astype("float32"),
        "dones": np.zeros(4, dtype="float32"),
        "steps": np.full(4, 3),
        "weights": np.ones(4, dtype="float32"),
    }
    result = agent.train_step(batch)
    assert result["td_errors"].shape == (4,)
    assert np.isfinite(result["critic_1_loss"])
