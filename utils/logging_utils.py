import logging
from pathlib import Path
from typing import Optional, Union

def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    enabled: bool = True,
) -> logging.Logger:
    """
    Returns a configured python `logging.Logger`.

    Args:
        name: Logger name (e.g., PPO_CarRacing_seed0).
        log_dir: Optional directory to save log file (<name>.log).
        level: Logging level (DEBUG, INFO, ...).
        enabled: Master switch. If False, uses NullHandler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers and not getattr(logger, "_is_custom_configured", False):
        # Clear existing handlers if they were not set by this function,
        # or if we want to reconfigure (e.g., for testing).
        # This simple check might need refinement if loggers are shared complexly.
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    if not logger.handlers: # Only add handlers if none exist or they were cleared
        if not enabled:
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.CRITICAL + 1) # Effectively disable
        else:
            logger.setLevel(level)
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Console Handler
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)

            # Optional File Handler
            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_dir / f"{name}.log")
                fh.setFormatter(fmt)
                logger.addHandler(fh)
        
        logger.propagate = False
        setattr(logger, "_is_custom_configured", True)
        
    return logger