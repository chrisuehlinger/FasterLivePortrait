"""
Entry point for running the spd_editor module directly with python -m.
"""
import sys
from typing import List, Optional

from spd_editor.main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))