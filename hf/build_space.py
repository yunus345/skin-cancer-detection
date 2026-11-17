"""Hugging Face Space'i bu depodan olusturur ve (istenirse) yukler.

Space'e yalnizca uygulamanin calismasi icin gereken dosyalar gider; agirliklar gitmez (app.py onlari
model deposundan, sabit 'v1' etiketiyle indirir). Space'te elle degisiklik yapilmaz: kaynak her zaman
bu depo, Space bu betikle yeniden uretilir.

Calistirma:
    python hf/build_space.py --out <klasor>            # sadece klasoru hazirla
    python hf/build_space.py --out <klasor> --upload   # hazirla ve Space'e yukle
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = 'emreozkn/skin-lesion-demo'
FILES = [
    'app.py',
    'src/__init__.py',
    'src/data/__init__.py', 'src/data/loader.py',
    'src/training/__init__.py', 'src/training/train.py',
    'src/inference/__init__.py', 'src/inference/predictor.py',
]


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for rel in FILES:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out / rel)
    shutil.copytree(ROOT / 'assets' / 'examples', out / 'assets' / 'examples')
    shutil.copy2(ROOT / 'hf' / 'space_readme.md', out / 'README.md')
    shutil.copy2(ROOT / 'hf' / 'space_requirements.txt', out / 'requirements.txt')
    print(f'Space klasoru hazir: {out}')


def main():
    parser = argparse.ArgumentParser(description='Hugging Face Space klasorunu olustur / yukle.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()

    out = Path(args.out)
    build(out)
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(SPACE_ID, repo_type='space', space_sdk='gradio', private=False, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=SPACE_ID, repo_type='space',
                          commit_message='Space build from GitHub repo (hf/build_space.py)')
        print(f'https://huggingface.co/spaces/{SPACE_ID}')


if __name__ == '__main__':
    main()
