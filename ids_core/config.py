"""Configuración compartida y ubicación de los recursos del proyecto."""
import logging
from pathlib import Path

LOG = logging.getLogger("IDS")
DATASET = "solarmainframe/ids-intrusion-csv"
ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"flowid", "sourceip", "srcip", "destinationip", "dstip", "timestamp",
           "sourceport", "srcport", "sourcefile", "index"}
META = {"__source", "__priority"}
