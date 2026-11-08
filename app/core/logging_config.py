"""Standard-library logging configuration."""

import logging

_HANDLER_NAME = "llm-inference-performance-lab-console"


def configure_logging(level: str = "INFO") -> None:
    """Configure one console handler and apply a validated logging level."""
    normalized_level = level.upper()
    numeric_level = logging.getLevelName(normalized_level)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported logging level: {level!r}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    handler = next(
        (candidate for candidate in root_logger.handlers if candidate.get_name() == _HANDLER_NAME),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
    handler.setLevel(numeric_level)
