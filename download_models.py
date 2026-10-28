"""Download pinned public CLIP models for offline/local classroom use."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent / 'toolkit' / 'models' / 'Xenova' / 'clip-vit-base-patch32'
REVISION = 'd15189d7028b43f1d3e65039190477f6af591c2a'
FILES = ['config.json', 'preprocessor_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
         'onnx/text_model_quantized.onnx', 'onnx/vision_model_quantized.onnx']

def main():
    manifest = {'model': 'Xenova/clip-vit-base-patch32', 'revision': REVISION, 'files': []}
    for name in FILES:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{REVISION}/{name}'
        if not target.exists():
            print(f'Downloading {name} ...', flush=True)
            temporary = target.with_suffix(target.suffix + '.partial')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(target)
        manifest['files'].append({'path': name, 'bytes': target.stat().st_size,
                                 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Ready. Model files are local; no image or prompt is uploaded.', flush=True)

if __name__ == '__main__':
    main()
