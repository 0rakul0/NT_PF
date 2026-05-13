from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ARTIGO = Path(__file__).resolve().parents[1]
FIGURES = ARTIGO / "figures"


def gerar_hdbscan_ruido() -> None:
    df = pd.read_csv(FIGURES / "hdbscan_ruido_por_representacao.csv")
    labels = {
        "integral": "Integral",
        "resumo": "Resumo",
        "hibrido": "Hibrido",
    }
    df["representacao"] = df["representacao"].map(labels).fillna(df["representacao"])
    df = df.sort_values("ruido", ascending=False)

    width, height = 1600, 820
    margin_l, margin_r, margin_t, margin_b = 150, 100, 170, 130
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 28)
    small = ImageFont.truetype(str(font_path), 24)
    tiny = ImageFont.truetype(str(font_path), 20)
    title = ImageFont.truetype(str(bold_path), 34)
    bold = ImageFont.truetype(str(bold_path), 26)

    draw.text((margin_l, 35), "HDBSCAN: grupos densos e ruído por representação textual", fill="#202020", font=title)
    draw.rectangle((margin_l, 95, margin_l + 22, 117), fill="#d95f5f")
    draw.text((margin_l + 32, 89), "Ruído", fill="#202020", font=small)
    draw.rectangle((margin_l + 150, 95, margin_l + 172, 117), fill="#5aa469")
    draw.text((margin_l + 182, 89), "Agrupadas", fill="#202020", font=small)

    max_total = int((df["ruido"] + df["agrupadas"]).max())
    for tick in range(0, max_total + 1, 2000):
        y = margin_t + plot_h - (tick / max_total) * plot_h
        draw.line((margin_l, y, width - margin_r, y), fill="#dddddd", width=1)
        draw.text((35, y - 12), f"{tick:,}".replace(",", "."), fill="#303030", font=tiny)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#303030", width=2)
    draw.line((margin_l, margin_t + plot_h, width - margin_r, margin_t + plot_h), fill="#303030", width=2)

    bar_w = 150
    gap = (plot_w - bar_w * len(df)) / (len(df) + 1)
    for i, row in enumerate(df.itertuples(index=False)):
        x0 = margin_l + gap * (i + 1) + bar_w * i
        x1 = x0 + bar_w
        total = int(row.ruido + row.agrupadas)
        ruido_h = row.ruido / max_total * plot_h
        agrup_h = row.agrupadas / max_total * plot_h
        y_base = margin_t + plot_h
        y_ruido = y_base - ruido_h
        y_agrup = y_ruido - agrup_h
        draw.rectangle((x0, y_ruido, x1, y_base), fill="#d95f5f")
        draw.rectangle((x0, y_agrup, x1, y_ruido), fill="#5aa469")
        pct = row.ruido / total * 100 if total else 0
        label = f"{pct:.1f}% ruído".replace(".", ",")
        draw.text((x0 + bar_w / 2, y_agrup - 36), label, fill="#202020", font=tiny, anchor="mm")
        draw.text((x0 + bar_w / 2, y_ruido + ruido_h / 2), f"{int(row.ruido):,}".replace(",", "."), fill="white", font=bold, anchor="mm")
        draw.text((x0 + bar_w / 2, y_base + 38), row.representacao, fill="#202020", font=small, anchor="mm")

    img.save(FIGURES / "hdbscan_ruido.png")


if __name__ == "__main__":
    gerar_hdbscan_ruido()
