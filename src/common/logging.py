"""loguru 기반 로깅 설정.

Layra는 stderr에 색상 로그, 선택적으로 파일에도 기록한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """loguru 전역 로거를 구성한다.

    여러 번 호출되어도 한 번만 설정한다.

    Args:
        level: 로그 레벨 (DEBUG / INFO / WARNING / ERROR).
        log_file: 지정되면 해당 파일에도 로그 기록.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
        )

    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """모듈 이름을 bind한 logger 반환."""
    from loguru import logger

    if not _CONFIGURED:
        setup_logging()
    return logger.bind(name=name)
