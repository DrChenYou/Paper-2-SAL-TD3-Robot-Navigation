import torch

from sal_td3.networks import MultimodalObservationEncoder, TemporalActor, TemporalCritic


def test_multimodal_encoder_and_network_shapes():
    encoder = MultimodalObservationEncoder(output_dim=64)
    state = encoder(torch.rand(2, 5, 360), torch.rand(2, 5, 4))
    actor = TemporalActor(64, hidden_dim=64, lstm_hidden_dim=32)
    critic = TemporalCritic(64, hidden_dim=64, lstm_hidden_dim=32)
    action = actor(state)
    value = critic(state, action)
    assert state.shape == (2, 5, 64)
    assert action.shape == (2, 2)
    assert torch.all((action >= 0) & (action <= 1))
    assert value.shape == (2, 1)
