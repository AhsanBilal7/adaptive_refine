"""Argument parser for dataclass-based configs with YAML support."""

import sys
import yaml
from dataclasses import fields
from typing import get_origin, get_args, Optional


class H4ArgumentParser:
    """
    Parse a dataclass Config from a YAML file + command-line overrides.

    Usage:
        parser = H4ArgumentParser(Config)
        config = parser.parse()
    """

    def __init__(self, config_class):
        self.config_class = config_class

    def parse(self):
        args = sys.argv[1:]

        # First positional arg is the YAML config file (if it exists and doesn't start with --)
        yaml_config = {}
        remaining_args = []
        if args and not args[0].startswith("--"):
            yaml_path = args[0]
            try:
                with open(yaml_path, "r") as f:
                    yaml_config = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(f"Warning: config file {yaml_path} not found, using defaults.")
            remaining_args = args[1:]
        else:
            remaining_args = args

        # Parse command-line overrides (--key=value format)
        cli_overrides = {}
        for arg in remaining_args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg[2:].split("=", 1)
                    cli_overrides[key] = value
                else:
                    key = arg[2:]
                    cli_overrides[key] = True

        # Merge: defaults < yaml < cli
        merged = {}
        for f in fields(self.config_class):
            name = f.name
            if name in cli_overrides:
                merged[name] = self._cast(cli_overrides[name], f.type)
            elif name in yaml_config:
                merged[name] = yaml_config[name]
            # else: use dataclass default

        return self.config_class(**merged)

    def _cast(self, value, type_hint):
        """Cast a string value to the appropriate type."""
        # Handle Optional[X]
        origin = get_origin(type_hint)
        if origin is type(Optional[str]):  # Union
            type_args = get_args(type_hint)
            # Optional[X] is Union[X, None]
            non_none = [t for t in type_args if t is not type(None)]
            if non_none:
                type_hint = non_none[0]

        if value is None or value == "None" or value == "none":
            return None

        if type_hint == bool or type_hint is bool:
            if isinstance(value, bool):
                return value
            return value.lower() in ("true", "1", "yes")
        elif type_hint == int or type_hint is int:
            return int(value)
        elif type_hint == float or type_hint is float:
            return float(value)
        elif type_hint == str or type_hint is str:
            return str(value)
        else:
            # Fallback: try the type as constructor
            try:
                return type_hint(value)
            except (TypeError, ValueError):
                return value
