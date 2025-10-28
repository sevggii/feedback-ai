"""
Logging konfigürasyonu ve yardımcı fonksiyonlar.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from .config import config

# Rich traceback'i etkinleştir
install()

# Rich console
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_handler: bool = True
) -> logging.Logger:
    """
    Logging'i yapılandır.
    
    Args:
        level: Log seviyesi
        log_file: Log dosyası yolu (opsiyonel)
        rich_handler: Rich handler kullanılsın mı
        
    Returns:
        Yapılandırılmış logger
    """
    # Log dizinini oluştur
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Logger'ı oluştur
    logger = logging.getLogger("social_feedback_ai")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Mevcut handler'ları temizle
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (Rich)
    if rich_handler:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    else:
        # Standart console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (opsiyonel)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Logger al.
    
    Args:
        name: Logger adı (opsiyonel)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"social_feedback_ai.{name}")
    return logging.getLogger("social_feedback_ai")


# Global logger
logger = setup_logging(
    level=config.LOG_LEVEL,
    log_file=config.LOG_FILE,
    rich_handler=True
)
