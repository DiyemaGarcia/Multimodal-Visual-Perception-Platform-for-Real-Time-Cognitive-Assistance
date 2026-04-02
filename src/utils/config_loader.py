import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Loads and merges YAML configuration files.
    Supports nested access via dot notation.
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML config file by name (without .yaml extension).
        Caches the result.
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def load_all(self) -> Dict[str, Any]:
        """
        Load all YAML files in the config directory and merge into one dict.
        """
        merged = {}
        for yaml_file in self.config_dir.glob("*.yaml"):
            name = yaml_file.stem
            merged[name] = self.load(name)
        return merged

    def get(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """
        Get a nested value using dot-style key access.
        Example: loader.get("model", "encoder", "hidden_dim")
        """
        config = self.load(config_name)
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def override(self, config_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override specific keys in a loaded config (shallow merge).
        Useful for CLI-style experiment overrides.
        """
        config = self.load(config_name).copy()
        config.update(overrides)
        self._configs[config_name] = config
        return config


def load_config(config_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Convenience function for one-shot config loading."""
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)