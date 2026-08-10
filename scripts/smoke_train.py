#!/usr/bin/env python3
"""Exercise SAL-TD3 optimization with deterministic synthetic histories."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from sal_td3 import SALTD3Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    agent = SALTD3Agent(args.state_dim, hidden_dim=64, lstm_hidden_dim=32)
    last = None
    for _ in range(args.steps):
        shape = (args.batch_size, args.sequence_length, args.state_dim)
        batch = {
            "states": rng.normal(size=shape).astype(np.float32),
            "actions": rng.uniform(size=(args.batch_size, 2)).astype(np.float32),
            "rewards": rng.normal(size=args.batch_size).astype(np.float32),
            "next_states": rng.normal(size=shape).astype(np.float32),
            "dones": rng.integers(0, 2, size=args.batch_size).astype(np.float32),
            "steps": np.full(args.batch_size, 3, dtype=np.int64),
            "weights": np.ones(args.batch_size, dtype=np.float32),
        }
        last = agent.train_step(batch)
    print(
        f"completed={args.steps} critic_1_loss={last['critic_1_loss']:.6f} "
        f"critic_2_loss={last['critic_2_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
