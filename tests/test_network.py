from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from alphazero.games.gomoku import GomokuGame
from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
    ResidualBlock,
)


@pytest.fixture
def config() -> NetworkConfig:
    return NetworkConfig(
        board_size=9,
        input_channels=3,
        channels=64,
        residual_blocks=5,
        value_hidden_channels=32,
        value_hidden_size=64,
    )


@pytest.fixture
def model(config: NetworkConfig) -> PolicyValueNet:
    return PolicyValueNet(config)


def test_network_config(config: NetworkConfig) -> None:
    assert config.board_size == 9
    assert config.input_channels == 3
    assert config.channels == 64
    assert config.residual_blocks == 5
    assert config.action_size == 81


def test_invalid_network_config() -> None:
    with pytest.raises(ValueError):
        PolicyValueNet(
            NetworkConfig(board_size=0)
        )

    with pytest.raises(ValueError):
        PolicyValueNet(
            NetworkConfig(input_channels=0)
        )

    with pytest.raises(ValueError):
        PolicyValueNet(
            NetworkConfig(channels=0)
        )

    with pytest.raises(ValueError):
        PolicyValueNet(
            NetworkConfig(residual_blocks=-1)
        )


def test_residual_block_shape() -> None:
    block = ResidualBlock(channels=32)
    x = torch.randn(4, 32, 9, 9)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_network_output_shapes(
    model: PolicyValueNet,
) -> None:
    batch_size = 4
    states = torch.randn(batch_size, 3, 9, 9)

    policy_logits, values = model(states)

    assert policy_logits.shape == (batch_size, 81)
    assert values.shape == (batch_size, 1)

    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()


def test_value_is_in_valid_range(
    model: PolicyValueNet,
) -> None:
    states = torch.randn(8, 3, 9, 9)
    _, values = model(states)

    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


def test_predict_returns_probabilities(
    model: PolicyValueNet,
) -> None:
    states = torch.randn(4, 3, 9, 9)

    policy_probs, values = model.predict(states)

    assert policy_probs.shape == (4, 81)
    assert values.shape == (4, 1)

    assert torch.all(policy_probs >= 0.0)
    assert torch.all(policy_probs <= 1.0)

    probability_sums = policy_probs.sum(dim=1)

    assert torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
    )


def test_predict_restores_training_mode(
    model: PolicyValueNet,
) -> None:
    model.train()
    states = torch.randn(2, 3, 9, 9)

    model.predict(states)

    assert model.training is True

    model.eval()
    model.predict(states)

    assert model.training is False


def test_network_accepts_encoded_game_state(
    model: PolicyValueNet,
) -> None:
    game = GomokuGame(board_size=9, connect=5)
    state = game.initial_state()

    state = game.next_state(
        state,
        game.coord_to_action(4, 4),
    )

    encoded = game.encode_state(state)

    tensor = torch.from_numpy(encoded).unsqueeze(0)

    policy_logits, value = model(tensor)

    assert policy_logits.shape == (1, 81)
    assert value.shape == (1, 1)


def test_backward_pass(
    model: PolicyValueNet,
) -> None:
    batch_size = 4

    states = torch.randn(
        batch_size,
        3,
        9,
        9,
    )

    target_policy = torch.full(
        (batch_size, 81),
        fill_value=1.0 / 81.0,
    )

    target_value = torch.tensor(
        [[1.0], [-1.0], [0.0], [1.0]],
        dtype=torch.float32,
    )

    policy_logits, predicted_value = model(states)

    log_policy = torch.log_softmax(
        policy_logits,
        dim=1,
    )

    policy_loss = -(
        target_policy * log_policy
    ).sum(dim=1).mean()

    value_loss = nn.functional.mse_loss(
        predicted_value,
        target_value,
    )

    loss = policy_loss + value_loss
    loss.backward()

    assert torch.isfinite(loss)

    parameters_with_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(parameters_with_gradients) > 0

    for parameter in parameters_with_gradients:
        assert torch.isfinite(parameter.grad).all()


def test_optimizer_step_changes_parameters(
    model: PolicyValueNet,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    states = torch.randn(4, 3, 9, 9)
    target_value = torch.zeros(4, 1)

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    policy_logits, predicted_value = model(states)

    policy_loss = (
        -torch.log_softmax(policy_logits, dim=1).mean()
    )

    value_loss = nn.functional.mse_loss(
        predicted_value,
        target_value,
    )

    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )

    assert changed


def test_save_and_load_state_dict(
    model: PolicyValueNet,
    config: NetworkConfig,
    tmp_path: Path,
) -> None:
    model.eval()

    states = torch.randn(2, 3, 9, 9)

    with torch.no_grad():
        original_policy, original_value = model(states)

    checkpoint_path = tmp_path / "network.pth"

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    loaded_model = PolicyValueNet(config)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    with torch.no_grad():
        loaded_policy, loaded_value = loaded_model(states)

    assert torch.allclose(
        original_policy,
        loaded_policy,
        atol=1e-6,
    )

    assert torch.allclose(
        original_value,
        loaded_value,
        atol=1e-6,
    )


def test_invalid_input_dimensions(
    model: PolicyValueNet,
) -> None:
    with pytest.raises(ValueError, match="四维张量"):
        model(torch.randn(3, 9, 9))

    with pytest.raises(ValueError, match="输入通道数"):
        model(torch.randn(2, 2, 9, 9))

    with pytest.raises(ValueError, match="输入高度"):
        model(torch.randn(2, 3, 8, 9))

    with pytest.raises(ValueError, match="输入宽度"):
        model(torch.randn(2, 3, 9, 8))

    with pytest.raises(TypeError, match="浮点张量"):
        model(
            torch.zeros(
                2,
                3,
                9,
                9,
                dtype=torch.int64,
            )
        )


def test_parameter_count(
    model: PolicyValueNet,
) -> None:
    parameter_count = model.parameter_count()

    assert parameter_count > 0
    assert parameter_count < 2_000_000


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA 不可用",
)
def test_cuda_forward_and_backward(
    config: NetworkConfig,
) -> None:
    device = torch.device("cuda")

    model = PolicyValueNet(config).to(device)
    model.train()

    states = torch.randn(
        8,
        3,
        9,
        9,
        device=device,
    )

    policy_logits, values = model(states)

    loss = (
        policy_logits.square().mean()
        + values.square().mean()
    )

    loss.backward()
    torch.cuda.synchronize()

    assert policy_logits.device.type == "cuda"
    assert values.device.type == "cuda"
    assert torch.isfinite(loss)
