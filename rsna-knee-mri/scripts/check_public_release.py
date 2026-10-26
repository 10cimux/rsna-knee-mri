"""Conservative release checks; not a substitute for manual privacy/licence review."""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
for path in root.rglob('*'):
    if not path.is_file() or any(p in {'.git','__pycache__','.venv','build','dist'} or p.endswith('.egg-info') for p in path.parts):
        continue
    assert path.suffix.lower() not in {'.dcm','.npz','.npy','.pt','.pth','.safetensors','.joblib'}, path
    assert path.stat().st_size < 500_000, path
    text=path.read_text(encoding='utf-8')
    assert not re.search(r'1\.2\.826\.0\.1\.3680043\.',text), path
    assert 'C:' + chr(92) + 'Users' not in text, path
    assert not re.search(r'(?i)(?:api_key|access_token|password)\s*[:=]\s*[\'"][^\'"]{12,}',text), path
print('PASS: data-file, size, known-UID, local-path and obvious-credential checks')
