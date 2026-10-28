"""Create a static build. Large model files stay on pinned Hugging Face URLs."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parent
destination = root / 'dist'
destination.mkdir(exist_ok=True)
for source in (root / 'toolkit').iterdir():
    if source.name == 'models':
        continue
    target = destination / source.name
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
print('Static build ready:', destination)
