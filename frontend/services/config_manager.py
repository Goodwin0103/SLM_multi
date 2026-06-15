import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "temp" / "last_config.json"


class ConfigManager:
    """Handles reading and writing of training/testing config JSON files."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load config from disk. Returns an empty dict if file does not exist."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save_config(self, config: Dict[str, Any]) -> None:
        """Persist config to disk."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def merge_with_defaults(
        self, user_config: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill missing keys in user_config from defaults."""
        merged = dict(defaults)
        merged.update(user_config)
        return merged
