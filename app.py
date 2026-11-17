"""Gradio arayüzü: final_v1 modeliyle cilt lezyonu sınıflandırma, Grad-CAM ile.

Araştırma ve öğrenme amaçlıdır, tanı aracı değildir. Model yalnızca models/<ad>/config.json ve
src.inference.Predictor üzerinden kullanılır; model değişince bu dosya değişmez.

Çalıştırma: python app.py  (önce: python -m src.inference.package_model)
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src.inference.predictor import Predictor

MODEL_DIR = os.environ.get('MODEL_DIR', 'models/final_v1')
# Yerelde paket yoksa (or. Hugging Face Space'te) model Hub'dan indirilir. Revizyon sabit bir etiket:
# depoya yeni surum yuklense bile demo hangi modeli kullandigini bilir; v2'ye gecis bilincli bir degisiklik olur.
MODEL_REPO = os.environ.get('MODEL_REPO', 'emreozkn/skin-lesion-convnext-v1')
MODEL_REVISION = os.environ.get('MODEL_REVISION', 'v1')
EXAMPLES_DIR = Path('assets/examples')

if not Path(MODEL_DIR, 'config.json').exists():
    from huggingface_hub import snapshot_download

    MODEL_DIR = snapshot_download(MODEL_REPO, revision=MODEL_REVISION)

CLASS_NAMES = {
    'akiec': 'Aktinik keratoz / Bowen (akiec)',
    'bcc': 'Bazal hücreli karsinom (bcc)',
    'bkl': 'Benign keratoz benzeri lezyon (bkl)',
    'df': 'Dermatofibrom (df)',
    'mel': 'Melanom (mel)',
    'nv': 'Melanositik nevüs, ben (nv)',
    'vasc': 'Vasküler lezyon (vasc)',
}
MALIGNANT = {'akiec', 'bcc', 'mel'}

predictor = Predictor(MODEL_DIR)
config = predictor.config
UNCERTAIN_BELOW = config.get('uncertainty', {}).get('threshold')
MODES = {'Normal (en olası sınıf)': 'normal',
         f'Tarama (melanom olasılığı ≥ {predictor.mel_threshold:.2f} ise melanom)': 'tarama'}
GRADCAM_DEFAULT = 'Kararın sınıfı'


def decision_text(result: dict, mode: str) -> str:
    label = result['decision'][mode]
    confidence = max(result['probs'].values())
    lines = [f'### Model kararı: {CLASS_NAMES[label]}']
    if mode == 'tarama' and label != result['decision']['normal']:
        lines.append(f'En olası sınıf {CLASS_NAMES[result["decision"]["normal"]]}, ama melanom olasılığı '
                     f'({result["probs"]["mel"]:.2f}) tarama eşiğini geçtiği için melanom olarak işaretlendi.')
    if label in MALIGNANT:
        lines.append('Bu sınıf kötü huylu ya da kötü huylu olabilen bir lezyon türüdür.')
    if UNCERTAIN_BELOW is not None and confidence < UNCERTAIN_BELOW:
        lines.append(f'**Model emin değil** (en yüksek olasılık {confidence:.2f} < {UNCERTAIN_BELOW:.2f}). '
                     'Bu görüntü için doktor incelemesi önerilir.')
    else:
        lines.append(f'En yüksek olasılık {confidence:.2f}. Model emin olduğunda da yanılabilir: validation '
                     f'setinde 102 melanomun {config["uncertainty"]["val"]["missed_mel_while_sure"]}\'ini '
                     'emin olduğu halde kaçırdı.' if UNCERTAIN_BELOW is not None else '')
    return '\n\n'.join(line for line in lines if line)


def analyze(image, mode_label, gradcam_choice):
    if image is None:
        return None, 'Bir görüntü yükleyin ya da aşağıdaki örneklerden birini seçin.', None
    mode = MODES[mode_label]
    result = predictor.predict(image)
    probs = {CLASS_NAMES[k]: v for k, v in result['probs'].items()}
    target = result['decision'][mode] if gradcam_choice == GRADCAM_DEFAULT else gradcam_choice
    heatmap, _ = predictor.gradcam(image, target)
    overlay = predictor.overlay(image, heatmap)
    return probs, decision_text(result, mode), gr.Image(value=overlay, label=f'Grad-CAM: {CLASS_NAMES[target]}')


def metrics_text() -> str:
    m = config['metrics']
    row = lambda name, d: (f'| {name} ({d["n"]} görüntü) | {d["macro_f1"]:.3f} | {d["balanced_accuracy"]:.3f} | '
                           f'{d["mel_recall"]:.3f} |')
    return '\n'.join([
        f'**Model:** {config["name"]} ({config["version"]}), iki ConvNeXt-Tiny modelinin kalibre ortalaması, '
        f'{config["image_size"]} px.',
        '',
        '| Set | macro F1 | balanced accuracy | melanom recall |',
        '|---|---|---|---|',
        row('Dış test, ISIC 2018', m['external_test']),
        row('İç test (iyimser)', m['internal_test']),
        '',
        'Genelleme için dış test sayısı esas alınmalı. Isı haritası yalnızca bir model (seed 42) üzerinde '
        'hesaplanır ve 12×12 çözünürlükte kaba bir haritadır; modelin "nereye baktığına" dair yaklaşık bir '
        'ipucudur, açıklama değildir.',
    ])


with gr.Blocks(title='Cilt Lezyonu Sınıflandırma (araştırma)') as demo:
    gr.Markdown(
        '# Cilt lezyonu sınıflandırma\n'
        '**Araştırma ve öğrenme amaçlıdır, tanı aracı değildir.** Gerçek hasta görüntüsü yüklemeyin. '
        'Model yalnızca dermoskopi görüntüleriyle eğitildi; telefon fotoğraflarında sonuçlar anlamsız olabilir.'
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label='Dermoskopi görüntüsü')
            mode = gr.Radio(list(MODES), value=list(MODES)[0], label='Çalışma modu')
            gradcam_class = gr.Dropdown([GRADCAM_DEFAULT] + predictor.labels, value=GRADCAM_DEFAULT,
                                        label='Isı haritası hangi sınıf için?')
            button = gr.Button('Analiz et', variant='primary')
        with gr.Column():
            decision = gr.Markdown()
            probs = gr.Label(num_top_classes=3, label='En olası 3 sınıf')
            heatmap = gr.Image(label='Grad-CAM')
    gr.Examples(
        examples=[[str(p)] for p in sorted(EXAMPLES_DIR.glob('*.jpg'))],
        inputs=[image],
        label='Örnekler (HAM10000 validation seti, dosya adı gerçek sınıf; CC BY-NC 4.0, ViDIR Group)',
    )
    with gr.Accordion('Model ve ölçülen başarı', open=False):
        gr.Markdown(metrics_text())

    button.click(analyze, inputs=[image, mode, gradcam_class], outputs=[probs, decision, heatmap])


if __name__ == '__main__':
    # SHARE=1: Gradio gecici bir herkese acik link verir (72 saat, yalnizca bu bilgisayar acikken calisir).
    demo.launch(share=os.environ.get('SHARE') == '1')
