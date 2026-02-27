import sys
from pathlib import Path

# Add deployment directory to path
sys.path.insert(0, str(Path(__file__).parent / 'deployment'))

from app import app

if __name__ == "__main__":
    app.run()
