from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


_REQUIRED_SECTIONS = {
    "experiment",
    "game",
    "network",
    "mcts",
    "self_play",
    "training",
    "arena",
}


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate a YAML experiment config."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Config file is empty: {path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Config root must be a mapping"
        )

    config = deepcopy(config)
    validate_config(config)
    return config


def validate_config(
    config: Mapping[str, Any],
) -> None:
    """Validate required AlphaZero configuration values."""
    if not isinstance(config, Mapping):
        raise TypeError(
            "config must be a mapping"
        )

    missing_sections = (
        _REQUIRED_SECTIONS - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Config is missing sections: "
            f"{sorted(missing_sections)}"
        )

    for section in _REQUIRED_SECTIONS:
        if not isinstance(
            config[section],
            Mapping,
        ):
            raise ValueError(
                f"Config section '{section}' "
                "must be a mapping"
            )

    _require_positive_int(
        config,
        "experiment.seed",
        allow_zero=True,
    )
    _require_positive_int(config, "game.board_size")
    _require_positive_int(config, "game.connect")
    _require_positive_int(
        config,
        "network.input_channels",
    )
    _require_positive_int(config, "network.channels")
    _require_positive_int(
        config,
        "network.residual_blocks",
        allow_zero=True,
    )
    _require_positive_int(
        config,
        "mcts.num_simulations",
    )
    _require_positive_number(config, "mcts.c_puct")
    _require_positive_int(
        config,
        "self_play.games_per_iteration",
    )
    _require_positive_int(
        config,
        "self_play.temperature_moves",
        allow_zero=True,
    )
    _require_positive_int(
        config,
        "training.batch_size",
    )
    _require_positive_int(
        config,
        "training.replay_capacity",
    )
    _require_positive_int(
        config,
        "training.steps_per_iteration",
    )
    _require_positive_number(
        config,
        "training.learning_rate",
    )
    _require_positive_int(config, "arena.games")
    _require_probability(
        config,
        "arena.promotion_threshold",
    )

    board_size = get_config_value(
        config,
        "game.board_size",
    )
    connect = get_config_value(
        config,
        "game.connect",
    )

    if connect > board_size:
        raise ValueError(
            "game.connect cannot exceed "
            "game.board_size"
        )

    input_channels = get_config_value(
        config,
        "network.input_channels",
    )

    if input_channels != 3:
        raise ValueError(
            "The current Gomoku encoder requires "
            "network.input_channels=3"
        )


def get_config_value(
    config: Mapping[str, Any],
    dotted_key: str,
) -> Any:
    """Read a nested value such as 'network.channels'."""
    current: Any = config

    for key in dotted_key.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(
                f"Config path is not a mapping: {dotted_key}"
            )

        if key not in current:
            raise KeyError(
                f"Missing config key: {dotted_key}"
            )

        current = current[key]

    return current


def set_config_value(
    config: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """Set an existing nested config value."""
    keys = dotted_key.split(".")
    current: dict[str, Any] = config

    for key in keys[:-1]:
        if key not in current:
            raise KeyError(
                f"Missing config key: {dotted_key}"
            )

        nested = current[key]

        if not isinstance(nested, dict):
            raise KeyError(
                f"Config path is not a mapping: {dotted_key}"
            )

        current = nested

    final_key = keys[-1]

    if final_key not in current:
        raise KeyError(
            f"Missing config key: {dotted_key}"
        )

    current[final_key] = value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """
    Apply CLI overrides such as:
        training.batch_size=128
        training.use_amp=false
    """
    updated = deepcopy(dict(config))

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                "Override must use key=value format: "
                f"{override}"
            )

        dotted_key, raw_value = override.split(
            "=",
            maxsplit=1,
        )

        dotted_key = dotted_key.strip()

        if not dotted_key:
            raise ValueError(
                "Override key must not be empty"
            )

        value = yaml.safe_load(raw_value)
        set_config_value(
            updated,
            dotted_key,
            value,
        )

    validate_config(updated)
    return updated


def _require_positive_int(
    config: Mapping[str, Any],
    dotted_key: str,
    *,
    allow_zero: bool = False,
) -> None:
    value = get_config_value(config, dotted_key)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(
            f"{dotted_key} must be an integer"
        )

    minimum = 0 if allow_zero else 1

    if value < minimum:
        raise ValueError(
            f"{dotted_key} must be >= {minimum}"
        )


def _require_positive_number(
    config: Mapping[str, Any],
    dotted_key: str,
) -> None:
    value = get_config_value(config, dotted_key)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{dotted_key} must be a number"
        )

    if value <= 0:
        raise ValueError(
            f"{dotted_key} must be greater than 0"
        )


def _require_probability(
    config: Mapping[str, Any],
    dotted_key: str,
) -> None:
    value = get_config_value(config, dotted_key)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{dotted_key} must be a number"
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{dotted_key} must be within [0, 1]"
        )
