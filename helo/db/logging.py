import logging
import sys
from typing import Optional

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


def create_logger() -> logging.Logger:
    logger = logging.getLogger("helo")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ColoredFormatter())

    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    return logger


class ColoredFormatter(logging.Formatter):
    LOG_FORMAT = "[$TCS%(asctime)s$TCN] [%(levelname)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    COLOR_SEQ = "\033[1;{}m"
    RESET_SEQ = "\033[0m"
    COLORS = {
        "WARNING": YELLOW,
        "DEBUG": BLUE,
        "INFO": GREEN,
        "ERROR": RED,
        "CRITICAL": MAGENTA,
    }

    def __init__(
        self, fmt: Optional[str] = None, datefmt: Optional[str] = None
    ) -> None:
        logging.Formatter.__init__(
            self, fmt or self.logformat, datefmt or self.DATE_FORMAT
        )

    @property
    def logformat(self) -> str:
        return (
            self.LOG_FORMAT
            .replace("$TCS", self.COLOR_SEQ.format(30 + CYAN))
            .replace("$TCN", self.RESET_SEQ)
        )

    def _to(self, color: int, string: str) -> str:
        return "{}{}{}".format(
            self.COLOR_SEQ.format(30 + color),
            string,
            self.RESET_SEQ,
        )

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = self._to(self.COLORS[levelname], levelname)
        return logging.Formatter.format(self, record)
