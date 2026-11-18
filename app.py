"""Gradio arayüzü: final_v1 modeliyle cilt lezyonu sınıflandırma, Grad-CAM ile.

Araştırma ve öğrenme amaçlıdır, tanı aracı değildir. Model yalnızca models/<ad>/config.json ve
src.inference.Predictor üzerinden kullanılır; model değişince bu dosya değişmez. Bu dosya yalnızca
görünümü kurar: karar mantığı (T, ortalama, eşikler) Predictor'da ve config.json'da.

Çalıştırma: python app.py  (önce: python -m src.inference.package_model; yoksa model HF Hub'dan iner)
"""
from __future__ import annotations

import html
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
    'akiec': 'Aktinik keratoz / Bowen',
    'bcc': 'Bazal hücreli karsinom',
    'bkl': 'Benign keratoz benzeri lezyon',
    'df': 'Dermatofibrom',
    'mel': 'Melanom',
    'nv': 'Melanositik nevüs (ben)',
    'vasc': 'Vasküler lezyon',
}
MALIGNANT = {'akiec', 'bcc', 'mel'}

predictor = Predictor(MODEL_DIR)
config = predictor.config
UNCERTAIN_BELOW = config.get('uncertainty', {}).get('threshold')
MEL_THRESHOLD = predictor.mel_threshold
MODES = {'Normal': 'normal', 'Tarama (melanom öncelikli)': 'tarama'}
GRADCAM_DEFAULT = 'Kararın sınıfı'
GRADCAM_CHOICES = [GRADCAM_DEFAULT] + [f'{CLASS_NAMES[k]} ({k})' for k in predictor.labels]

CSS = """
.app-wrap { max-width: 1180px; margin: 0 auto; }
.hero { padding: 22px 26px; border-radius: 16px; margin-bottom: 6px;
        background: linear-gradient(135deg, rgba(13,148,136,.16), rgba(99,102,241,.14));
        border: 1px solid var(--border-color-primary); }
.hero h1 { margin: 0 0 6px 0; font-size: 1.7rem; letter-spacing: -.01em; }
.hero p { margin: 0; opacity: .85; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.badge { font-size: .78rem; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border-color-primary);
         background: var(--block-background-fill); }
.badge b { font-weight: 650; }
.notice { margin-top: 14px; padding: 10px 14px; border-radius: 10px; font-size: .88rem;
          background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.45); }
.card { padding: 18px 20px; border-radius: 14px; border: 1px solid var(--border-color-primary);
        background: var(--block-background-fill); border-left-width: 6px; }
.card.high { border-left-color: #e11d48; }
.card.low { border-left-color: #10b981; }
.card.empty { border-left-color: var(--border-color-primary); opacity: .75; }
.eyebrow { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7; }
.decision { font-size: 1.6rem; font-weight: 700; margin: 4px 0 2px 0; }
.code { font-family: ui-monospace, monospace; font-size: .85rem; opacity: .7; }
.pill { display: inline-block; margin-top: 10px; font-size: .78rem; padding: 3px 10px; border-radius: 999px; }
.pill.high { background: rgba(225,29,72,.14); color: #e11d48; }
.pill.low { background: rgba(16,185,129,.14); color: #059669; }
.note { margin-top: 12px; font-size: .88rem; line-height: 1.45; }
.unsure { margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: .88rem;
          background: rgba(245,158,11,.13); border: 1px solid rgba(245,158,11,.5); }
.bars { padding: 14px 16px; border-radius: 14px; border: 1px solid var(--border-color-primary);
        background: var(--block-background-fill); }
.bars h4 { margin: 0 0 10px 0; font-size: .85rem; opacity: .8; font-weight: 600; }
.bar-row { display: grid; grid-template-columns: 190px 1fr 48px; align-items: center; gap: 10px; margin: 7px 0;
           font-size: .85rem; }
.bar-row.top { font-weight: 650; }
.track { position: relative; height: 10px; border-radius: 999px; background: var(--border-color-primary); }
.fill { height: 100%; border-radius: 999px; }
.fill.high { background: #e11d48; } .fill.low { background: #10b981; }
.marker { position: absolute; top: -4px; width: 2px; height: 18px; background: #f59e0b; }
.pct { text-align: right; font-variant-numeric: tabular-nums; }
.legend { margin-top: 10px; font-size: .75rem; opacity: .7; }
@media (max-width: 640px) { .bar-row { grid-template-columns: 120px 1fr 44px; } }
"""

THEME = gr.themes.Soft(primary_hue='teal', secondary_hue='indigo', neutral_hue='slate')


def risk(label: str) -> str:
    return 'high' if label in MALIGNANT else 'low'


def hero_html() -> str:
    ext = config['metrics']['external_test']
    return f"""
<div class="hero">
  <h1>Cilt lezyonu sınıflandırma</h1>
  <p>Dermoskopi görüntüsünden 7 lezyon türü. HAM10000 ile eğitilmiş iki ConvNeXt-Tiny modelinin kalibre ortalaması.</p>
  <div class="badges">
    <span class="badge">Model <b>{html.escape(config['name'])} ({html.escape(config['version'])})</b></span>
    <span class="badge">Dış test macro F1 <b>{ext['macro_f1']:.3f}</b></span>
    <span class="badge">Dış test BA <b>{ext['balanced_accuracy']:.3f}</b></span>
    <span class="badge">ISIC 2018, <b>{ext['n']}</b> görüntü</span>
  </div>
  <div class="notice"><b>Araştırma ve öğrenme amaçlıdır, tanı aracı değildir.</b> Gerçek hasta görüntüsü yüklemeyin.
  Model yalnızca dermoskopi görüntüleriyle eğitildi; telefon fotoğraflarında sonuçlar anlamsızdır.</div>
</div>"""


def empty_card() -> str:
    return ('<div class="card empty"><div class="eyebrow">Model kararı</div>'
            '<div class="note">Bir görüntü yükleyin ya da aşağıdaki örneklerden birini seçin.</div></div>')


def decision_card(result: dict, mode: str) -> str:
    label = result['decision'][mode]
    normal = result['decision']['normal']
    confidence = max(result['probs'].values())
    group = risk(label)
    group_text = 'Kötü huylu olabilen lezyon türü' if group == 'high' else 'İyi huylu lezyon türü'
    parts = [
        f'<div class="card {group}">',
        f'<div class="eyebrow">Model kararı · {"Tarama" if mode == "tarama" else "Normal"} modu</div>',
        f'<div class="decision">{html.escape(CLASS_NAMES[label])}</div>',
        f'<div class="code">{label} · en yüksek olasılık {confidence:.2f}</div>',
        f'<span class="pill {group}">{group_text}</span>',
    ]
    if mode == 'tarama' and label != normal:
        parts.append(f'<div class="note">En olası sınıf <b>{html.escape(CLASS_NAMES[normal])}</b>, ama melanom olasılığı '
                     f'({result["probs"]["mel"]:.2f}) tarama eşiğini ({MEL_THRESHOLD:.2f}) geçtiği için melanom '
                     'olarak işaretlendi.</div>')
    if UNCERTAIN_BELOW is not None and confidence < UNCERTAIN_BELOW:
        parts.append(f'<div class="unsure"><b>Model emin değil</b> (en yüksek olasılık {confidence:.2f} &lt; '
                     f'{UNCERTAIN_BELOW:.2f}). Bu görüntü için doktor incelemesi önerilir.</div>')
    elif UNCERTAIN_BELOW is not None:
        missed = config['uncertainty']['val']['missed_mel_while_sure']
        parts.append(f'<div class="note">Model emin olduğunda da yanılabilir: validation setinde 102 melanomun '
                     f'{missed}\'ini emin olduğu halde kaçırdı.</div>')
    parts.append('</div>')
    return ''.join(parts)


def bars_html(result: dict, mode: str) -> str:
    probs = result['probs']
    top = max(probs, key=probs.get)
    rows = []
    for label in sorted(probs, key=probs.get, reverse=True):
        p = probs[label]
        marker = ''
        if label == 'mel' and mode == 'tarama':
            marker = f'<div class="marker" style="left:{100 * MEL_THRESHOLD:.1f}%" title="tarama eşiği"></div>'
        rows.append(
            f'<div class="bar-row{" top" if label == top else ""}">'
            f'<div>{html.escape(CLASS_NAMES[label])} <span class="code">({label})</span></div>'
            f'<div class="track"><div class="fill {risk(label)}" style="width:{100 * p:.1f}%"></div>{marker}</div>'
            f'<div class="pct">{100 * p:.0f}%</div></div>')
    legend = ('Kırmızı: kötü huylu olabilen tür, yeşil: iyi huylu. '
              + (f'Turuncu çizgi: melanom tarama eşiği ({MEL_THRESHOLD:.2f}).' if mode == 'tarama' else ''))
    return f'<div class="bars"><h4>Sınıf olasılıkları (kalibre, iki modelin ortalaması)</h4>{"".join(rows)}' \
           f'<div class="legend">{legend}</div></div>'


def analyze(image, mode_label, gradcam_choice):
    if image is None:
        return empty_card(), '', None
    mode = MODES[mode_label]
    result = predictor.predict(image)
    if gradcam_choice == GRADCAM_DEFAULT:
        target = result['decision'][mode]
    else:
        target = gradcam_choice.rsplit('(', 1)[1].rstrip(')')
    heatmap, _ = predictor.gradcam(image, target)
    overlay = predictor.overlay(image, heatmap)
    slider = gr.ImageSlider(value=(image.convert('RGB'), overlay),
                            label=f'Orijinal ↔ Grad-CAM: {CLASS_NAMES[target]} ({target})')
    return decision_card(result, mode), bars_html(result, mode), slider


def about_md() -> str:
    m = config['metrics']
    row = lambda name, d: (f'| {name} | {d["n"]} | {d["macro_f1"]:.3f} | {d["balanced_accuracy"]:.3f} | '
                           f'{d["mel_recall"]:.3f} |')
    return '\n'.join([
        '| Set | Görüntü | macro F1 | balanced accuracy | melanom recall |',
        '|---|---|---|---|---|',
        row('Dış test, ISIC 2018', m['external_test']),
        row('İç test (iyimser)', m['internal_test']),
        '',
        '- **Normal mod:** en olası sınıf. **Tarama modu:** ortalama melanom olasılığı '
        f'{MEL_THRESHOLD:.2f} ve üstündeyse melanom; daha çok melanom yakalar, daha çok yanlış alarm verir.',
        f'- **"Emin değil" uyarısı:** en yüksek olasılık {UNCERTAIN_BELOW:.2f} altındaysa (validation setinde seçildi).',
        '- **Isı haritası (Grad-CAM):** yalnızca bir model (seed 42) üzerinde, 12×12 çözünürlükte kaba bir harita; '
        'modelin "nereye baktığına" dair yaklaşık bir ipucu, açıklama değildir.',
        '- Örnek görüntüler: HAM10000 validation seti (CC BY-NC 4.0, ViDIR Group, Medical University of Vienna).',
        '- Kod ve sınırlamalar: [GitHub](https://github.com/yunus345/skin-cancer-detection) · '
        f'Model: [Hugging Face]({"https://huggingface.co/" + MODEL_REPO})',
    ])


example_files = sorted(EXAMPLES_DIR.glob('*.jpg'))

with gr.Blocks(title='Cilt Lezyonu Sınıflandırma (araştırma)') as demo:
    with gr.Column(elem_classes='app-wrap'):
        gr.HTML(hero_html())
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                # PRELOAD_EXAMPLE=<dosya adi parcasi>: sayfa o ornekle acilir (ekran goruntusu/sunum icin).
                preload = next((str(p) for p in example_files if os.environ.get('PRELOAD_EXAMPLE', '\0') in p.name), None)
                image = gr.Image(value=preload, type='pil', label='Dermoskopi görüntüsü', height=340)
                mode = gr.Radio(list(MODES), value=os.environ.get('PRELOAD_MODE', 'Normal'), label='Çalışma modu',
                                info=f'Tarama: melanom olasılığı ≥ {MEL_THRESHOLD:.2f} ise melanom der.')
                gradcam_class = gr.Dropdown(GRADCAM_CHOICES, value=GRADCAM_DEFAULT,
                                            label='Isı haritası hangi sınıf için?')
                gr.Examples(
                    examples=[[str(p)] for p in example_files],
                    inputs=[image],
                    label='Örnekler (HAM10000 validation seti; sırasıyla gerçek sınıf: akiec, bcc, bkl, df, mel, nv, vasc)',
                )
            with gr.Column(scale=6):
                decision = gr.HTML(empty_card())
                probs = gr.HTML()
                heatmap = gr.ImageSlider(label='Orijinal ↔ Grad-CAM', type='pil', height=340)
        with gr.Accordion('Model, ölçülen başarı ve nasıl çalışır', open=False):
            gr.Markdown(about_md())

    inputs, outputs = [image, mode, gradcam_class], [decision, probs, heatmap]
    # Buton yok: goruntu, mod ya da isi haritasi sinifi degisince sonuc kendiliginden yenilenir.
    demo.load(analyze, inputs=inputs, outputs=outputs, api_name=False)
    for component in (image, mode, gradcam_class):
        component.change(analyze, inputs=inputs, outputs=outputs, api_name='analyze' if component is image else False)


if __name__ == '__main__':
    # SHARE=1: Gradio gecici bir herkese acik link verir (72 saat, yalnizca bu bilgisayar acikken calisir).
    demo.launch(share=os.environ.get('SHARE') == '1', theme=THEME, css=CSS)
