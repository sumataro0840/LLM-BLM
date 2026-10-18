"""
総合報告書 2 の図を、作られた場所から集める（図そのものは各実験のスクリプトが作る）。

    python 資料集/01_総合報告書/総合報告2/LLM-BLM_総合報告書2_図/作成スクリプト/collect.py

出どころ
- h0x_*.pdf, cumulative.pdf : 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験結果レポート_図/（作成スクリプト/figs.py）
- timing, quintile, car, compare, forest : 資料集/02_実験/実験2_決算発表文/図/（作成スクリプト/figs.py）
- x01_*.png : experiments/X01_covariance_denoise/results/figures/（figs.py）
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())  # リポジトリ（資料集の中で動かしても変わらない）
SOURCES = {
    REPO / '資料集' / '02_実験' / '実験1_仮説の検証' / 'LLM-BLM_実験結果レポート_図': ['h01_slsqp.pdf', 'h02_ic.pdf', 'h03_beta.pdf', 'h04_placebo.pdf',
                                                   'h05_trace.pdf', 'h06_ic.pdf', 'h07_feedback.pdf', 'cumulative.pdf'],
    REPO / '資料集' / '02_実験' / '実験2_決算発表文' / '図': ['timing.pdf', 'quintile.pdf', 'car.pdf', 'compare.pdf', 'forest.pdf'],
    REPO / 'experiments' / 'X01_covariance_denoise' / 'results' / 'figures': ['1_noise_only.png', '2_real_eigenvalues.png',
                                                                             '3_spectrum.png', '4_outcome.png'],
}

for folder, names in SOURCES.items():
    for name in names:
        target = HERE / (f'x01_{name}' if name.endswith('.png') else name)
        shutil.copy(folder / name, target)
        print(target.name)
