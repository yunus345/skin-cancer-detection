"""Paketlenmis bir modeli (package_model.py ciktisi) yukleyip tek goruntu icin tahmin uretir.

Arayuz yalnizca bu sinifi kullanir; egitim kodunun geri kalanina bagli degildir. On isleme egitimdeki
fonksiyonun kendisidir (build_transforms), kopyasi degil: ikisi ayrisirsa model sessizce kotulesir.

Ornek:
    predictor = Predictor('models/final_v1')
    result = predictor.predict(Image.open('lezyon.jpg'))
    result['decision']['normal'], result['top3']
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.training.train import build_model, build_transforms, select_device


class Predictor:
    def __init__(self, model_dir: str | Path, device: str = 'auto', verify_md5: bool = True):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / 'config.json').read_text())
        self.labels = self.config['labels']
        self.device = select_device(device)
        self.transform = build_transforms(self.config['image_size'], train=False)

        self.models, self.temperatures = {}, {}
        for member in self.config['members']:
            path = self.model_dir / member['weights']
            if verify_md5 and hashlib.md5(path.read_bytes()).hexdigest() != member['md5']:
                raise ValueError(f'{path} MD5 config.json ile uyusmuyor (dosya degismis veya bozuk)')
            model = build_model(num_classes=len(self.labels), architecture=self.config['architecture'],
                                pretrained=False)
            # weights_only=True: dosyada yalnizca tensor olabilir, kod calistirilamaz.
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            self.models[member['name']] = model.to(self.device).eval()
            self.temperatures[member['name']] = member['temperature']

        tarama = self.config['operating_points']['tarama']
        self.mel_idx = self.labels.index(tarama['class'])
        self.mel_threshold = tarama['threshold']

    @torch.no_grad()
    def predict_tensor(self, images: torch.Tensor) -> tuple[np.ndarray, dict]:
        """(N, 3, H, W) on islenmis tensor -> (ortalama olasiliklar (N, K), uye bazinda olasiliklar)."""
        images = images.to(self.device)
        member_probs = {}
        for name, model in self.models.items():
            logits = model(images).float().cpu().numpy()
            # Temperature scaling: logit / T, sonra softmax. T < 1 olasiliklari keskinlestirir.
            scaled = logits / self.temperatures[name]
            scaled -= scaled.max(axis=1, keepdims=True)
            exp = np.exp(scaled)
            member_probs[name] = exp / exp.sum(axis=1, keepdims=True)
        return np.mean(list(member_probs.values()), axis=0), member_probs

    def decide(self, probs: np.ndarray) -> dict:
        """Iki calisma noktasi. Tarama: mel olasiligi esigi gecerse baska sinif daha olasi olsa da mel."""
        normal = int(probs.argmax())
        tarama = self.mel_idx if probs[self.mel_idx] >= self.mel_threshold else normal
        return {'normal': self.labels[normal], 'tarama': self.labels[tarama]}

    def predict(self, pil_image) -> dict:
        tensor = self.transform(pil_image.convert('RGB')).unsqueeze(0)
        probs, member_probs = self.predict_tensor(tensor)
        probs = probs[0]
        order = probs.argsort()[::-1]
        return {
            'probs': {label: float(p) for label, p in zip(self.labels, probs)},
            'top3': [(self.labels[i], float(probs[i])) for i in order[:3]],
            'decision': self.decide(probs),
            'member_probs': {name: p[0].tolist() for name, p in member_probs.items()},
            'version': self.config['version'],
        }
