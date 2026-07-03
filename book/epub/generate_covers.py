from __future__ import annotations

import math
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "covers"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1600, 2400

FONT_DIR = Path.home() / "Library" / "Fonts"
FONT_REG = FONT_DIR / "NanumGothic-Regular.ttf"
FONT_BOLD = FONT_DIR / "NanumGothic-Bold.ttf"
FONT_EXTRA = FONT_DIR / "NanumGothic-ExtraBold.ttf"
FONT_MONO = FONT_DIR / "NanumGothicCoding-Regular.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


F_TITLE = font(FONT_EXTRA, 104)
F_TITLE_SMALL = font(FONT_EXTRA, 84)
F_SUB = font(FONT_BOLD, 42)
F_BODY = font(FONT_REG, 34)
F_BODY_BOLD = font(FONT_BOLD, 34)
F_MONO = font(FONT_MONO, 30)
F_MONO_SMALL = font(FONT_MONO, 25)
F_AUTHOR = font(FONT_BOLD, 46)


SLATE = (34, 48, 62)
SLATE_DARK = (21, 30, 39)
SLATE_MID = (77, 102, 126)
SLATE_LIGHT = (213, 222, 230)
PAPER = (247, 249, 250)
BLUE = (81, 122, 158)
BLUE_DARK = (46, 78, 109)
GREEN = (72, 145, 126)
YELLOW = (245, 177, 66)
ORANGE = (224, 124, 72)
INK = (28, 36, 45)
MUTED = (93, 107, 121)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw: ImageDraw.ImageDraw, y: int, text: str, fnt, fill, width: int = W) -> int:
    tw, th = text_size(draw, text, fnt)
    draw.text(((width - tw) / 2, y), text, font=fnt, fill=fill)
    return y + th


def wrap_lines(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(wrap(part, max_chars) or [""])
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill, max_chars: int, leading: int) -> int:
    x, y = xy
    for line in wrap_lines(text, max_chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += leading
    return y


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def add_common_footer(draw: ImageDraw.ImageDraw, dark: bool = False):
    fill = PAPER if dark else INK
    muted = (184, 195, 207) if dark else MUTED
    draw.text((120, H - 245), "황윤구 · 유창민 지음", font=F_AUTHOR, fill=fill)
    draw.text((120, H - 180), "PYTHON · COLAB · HUGGINGFACE", font=F_MONO, fill=muted)
    draw.text((W - 318, H - 180), "Ch 1–34", font=F_MONO, fill=muted)


def add_footer_panel(draw: ImageDraw.ImageDraw, dark: bool = False):
    if dark:
        rounded_rect(draw, (92, H - 285, W - 92, H - 115), 20, (5, 12, 22, 132), None, 0)
        add_common_footer(draw, dark=True)
    else:
        rounded_rect(draw, (92, H - 285, W - 92, H - 115), 20, (255, 255, 255, 184), None, 0)
        add_common_footer(draw, dark=False)


def cover_crop(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    target_ratio = W / H
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))
    return image.resize((W, H), Image.Resampling.LANCZOS)


def vertical_gradient(size: tuple[int, int], top_rgba, bottom_rgba) -> Image.Image:
    w, h = size
    layer = Image.new("RGBA", size)
    pix = layer.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        color = tuple(int(top_rgba[i] * (1 - t) + bottom_rgba[i] * t) for i in range(4))
        for x in range(w):
            pix[x, y] = color
    return layer


def draw_title_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    dark: bool,
    accent=(245, 177, 66),
):
    title_fill = PAPER if dark else INK
    sub_fill = (207, 221, 232) if dark else SLATE
    muted = (165, 184, 202) if dark else MUTED
    draw.text((x, y), "Hugging Face로", font=F_TITLE_SMALL, fill=title_fill)
    draw.text((x, y + 118), "시작하는", font=F_TITLE_SMALL, fill=title_fill)
    draw.text((x, y + 236), "텍스트 분석 입문", font=F_TITLE, fill=title_fill)
    draw.rectangle((x, y + 392, x + 300, y + 406), fill=accent)
    draw_wrapped(
        draw,
        (x, y + 466),
        "sklearn에서 BERT, GPT/LLM, Diffusion LM까지 이어지는 실습형 텍스트 분석 원고",
        F_SUB,
        sub_fill,
        27,
        58,
    )
    draw.text((x, y + 705), "Loss · Head · Tokenizer · Alignment", font=F_MONO, fill=muted)


def cover_generated_background(
    source: Path,
    target: Path,
    *,
    mode: str,
    accent=(245, 177, 66),
):
    img = cover_crop(Image.open(source)).convert("RGBA")
    if mode == "dark":
        overlay = Image.new("RGBA", (W, H), (4, 10, 18, 78))
        img = Image.alpha_composite(img, overlay)
        left = Image.new("RGBA", (880, H), (5, 12, 22, 0))
        left.alpha_composite(vertical_gradient((880, H), (5, 12, 22, 238), (5, 12, 22, 58)))
        img.alpha_composite(left, (0, 0))
        draw = ImageDraw.Draw(img)
        draw_title_block(draw, 108, 168, dark=True, accent=accent)
        add_footer_panel(draw, dark=True)
    elif mode == "card":
        shade = Image.new("RGBA", (W, H), (255, 255, 255, 36))
        img = Image.alpha_composite(img, shade)
        draw = ImageDraw.Draw(img)
        rounded_rect(draw, (88, 112, 1050, 1055), 36, (255, 255, 255, 224), (223, 230, 237), 3)
        draw_title_block(draw, 140, 190, dark=False, accent=accent)
        add_footer_panel(draw, dark=False)
    else:
        overlay = Image.new("RGBA", (W, H), (5, 12, 22, 42))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        rounded_rect(draw, (90, 130, 1040, 1010), 34, (10, 22, 36, 178), (110, 137, 160), 2)
        draw_title_block(draw, 140, 205, dark=True, accent=accent)
        add_footer_panel(draw, dark=True)
    img.convert("RGB").save(target, quality=95)


def cover_slate_grid():
    img = Image.new("RGB", (W, H), SLATE_DARK)
    draw = ImageDraw.Draw(img)
    for x in range(-200, W + 200, 88):
        draw.line((x, 0, x + 520, H), fill=(31, 44, 57), width=2)
    for y in range(120, H, 120):
        draw.line((0, y, W, y), fill=(33, 47, 61), width=1)
    for i in range(0, 42):
        x = 120 + (i * 117) % 1340
        y = 220 + (i * 173) % 1420
        r = 3 + (i % 4)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(83, 119, 150))

    draw.text((118, 168), "HUGGING FACE TEXT ANALYSIS", font=F_MONO, fill=(142, 165, 185))
    draw.text((118, 350), "Hugging Face로", font=F_TITLE_SMALL, fill=PAPER)
    draw.text((118, 470), "시작하는", font=F_TITLE_SMALL, fill=PAPER)
    draw.text((118, 590), "텍스트 분석 입문", font=F_TITLE, fill=(236, 242, 247))
    draw.rectangle((118, 760, 420, 772), fill=YELLOW)
    draw_wrapped(
        draw,
        (118, 830),
        "sklearn에서 BERT, GPT/LLM, Diffusion LM까지 이어지는 실습형 텍스트 분석 원고",
        F_SUB,
        (210, 222, 233),
        28,
        58,
    )

    axes = [
        ("MODEL", "sklearn → BERT → GPT"),
        ("TASK", "classification → generation"),
        ("LOSS", "MSE/BCE/CE → DPO/GRPO"),
        ("TOKEN", "TF-IDF → WordPiece → BPE"),
    ]
    y = 1180
    for i, (k, v) in enumerate(axes):
        x = 118 if i % 2 == 0 else 820
        yy = y + (i // 2) * 170
        rounded_rect(draw, (x, yy, x + 610, yy + 118), 22, (39, 56, 72), (91, 122, 150), 2)
        draw.text((x + 28, yy + 24), k, font=F_MONO_SMALL, fill=YELLOW)
        draw.text((x + 28, yy + 62), v, font=F_MONO_SMALL, fill=(219, 228, 235))

    add_common_footer(draw, dark=True)
    img.save(OUT / "cover-slate-grid.png", quality=95)


def cover_phase_axis():
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 520), fill=(232, 238, 243))
    draw.rectangle((0, 520, W, H), fill=PAPER)
    draw.text((115, 120), "Hugging Face로", font=F_TITLE_SMALL, fill=INK)
    draw.text((115, 238), "시작하는", font=F_TITLE_SMALL, fill=INK)
    draw.text((115, 356), "텍스트 분석 입문", font=F_TITLE, fill=BLUE_DARK)
    draw.text((118, 585), "Loss와 Head에서 GPT/LLM 정렬까지", font=F_SUB, fill=SLATE)

    phases = [
        ("PHASE 0", "TF-IDF · Loss 감각"),
        ("PHASE 1", "BERT task reformulation"),
        ("PHASE 2", "Korean BERT revisit"),
        ("PHASE 3", "Tokenizer · small BERT"),
        ("PHASE 4", "GPT/LLM · SFT · DPO · GRPO"),
        ("PHASE 5", "Diffusion LM"),
    ]
    x0, y0 = 185, 810
    draw.line((x0 + 28, y0 + 20, x0 + 28, y0 + 820), fill=SLATE_LIGHT, width=8)
    for i, (p, desc) in enumerate(phases):
        y = y0 + i * 145
        color = [BLUE, GREEN, YELLOW, ORANGE, SLATE_MID, BLUE_DARK][i]
        draw.ellipse((x0, y, x0 + 56, y + 56), fill=color)
        draw.text((x0 + 95, y - 5), p, font=F_MONO, fill=color)
        draw.text((x0 + 95, y + 43), desc, font=F_BODY_BOLD, fill=INK)
        draw.line((x0 + 95, y + 94, W - 180, y + 94), fill=(224, 230, 235), width=2)

    add_common_footer(draw, dark=False)
    img.save(OUT / "cover-phase-axis.png", quality=95)


def cover_token_river():
    img = Image.new("RGB", (W, H), (239, 244, 247))
    draw = ImageDraw.Draw(img)
    for i in range(32):
        y = 250 + i * 54
        offset = int(math.sin(i * 0.55) * 90)
        color = [BLUE, GREEN, YELLOW, ORANGE][i % 4]
        draw.rounded_rectangle((140 + offset, y, 1450 + offset, y + 26), radius=13, fill=(*color,))
        draw.rounded_rectangle((250 + offset, y + 10, 1080 + offset, y + 18), radius=4, fill=(255, 255, 255))

    rounded_rect(draw, (95, 125, 1505, 1005), 32, (255, 255, 255), (221, 229, 236), 3)
    draw.text((145, 205), "Hugging Face로", font=F_TITLE_SMALL, fill=INK)
    draw.text((145, 330), "시작하는", font=F_TITLE_SMALL, fill=INK)
    draw.text((145, 455), "텍스트 분석 입문", font=F_TITLE, fill=BLUE_DARK)
    draw.rectangle((148, 645, 590, 660), fill=ORANGE)
    draw_wrapped(
        draw,
        (145, 720),
        "토큰이 벡터가 되고, 벡터가 Loss를 지나, 언어모델의 행동으로 이어지는 전체 흐름",
        F_SUB,
        MUTED,
        28,
        58,
    )
    draw.text((145, 1135), "TF-IDF → WordPiece → BPE → CausalLM → Alignment", font=F_MONO, fill=SLATE)

    add_common_footer(draw, dark=False)
    img.save(OUT / "cover-token-river.png", quality=95)


def cover_minimal_axis():
    img = Image.new("RGB", (W, H), (250, 251, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 80, H), fill=BLUE_DARK)
    draw.rectangle((80, 0, 104, H), fill=YELLOW)
    draw.text((175, 180), "TEXT ANALYSIS", font=F_MONO, fill=MUTED)
    draw.text((175, 310), "Hugging Face로", font=F_TITLE_SMALL, fill=INK)
    draw.text((175, 430), "시작하는", font=F_TITLE_SMALL, fill=INK)
    draw.text((175, 550), "텍스트 분석 입문", font=F_TITLE, fill=INK)
    draw_wrapped(draw, (178, 740), "sklearn · BERT · GPT/LLM · Diffusion LM", F_SUB, BLUE_DARK, 24, 58)

    axis = [
        ("01", "MODEL", "작은 모델에서 큰 언어모델로"),
        ("02", "TASK", "분류에서 생성과 정렬로"),
        ("03", "LOSS", "MSE/BCE/CE에서 DPO/GRPO로"),
        ("04", "TOKEN", "TF-IDF에서 BPE까지"),
        ("05", "DATA", "영어와 한국어를 함께 추적"),
    ]
    y = 1060
    for i, (num, k, v) in enumerate(axis):
        yy = y + i * 145
        draw.text((180, yy), num, font=F_MONO, fill=YELLOW)
        draw.text((275, yy), k, font=F_MONO, fill=BLUE_DARK)
        draw.text((510, yy), v, font=F_BODY, fill=INK)
        draw.line((180, yy + 62, 1360, yy + 62), fill=SLATE_LIGHT, width=2)

    add_common_footer(draw, dark=False)
    img.save(OUT / "cover-minimal-axis.png", quality=95)


def cover_ai_illustrations():
    bg_dir = OUT / "generated-bg"
    candidates = [
        (bg_dir / "bg-token-core.png", OUT / "cover-illustration-token-core.png", "dark", YELLOW),
        (bg_dir / "bg-model-engine.png", OUT / "cover-illustration-model-engine.png", "glass", GREEN),
        (bg_dir / "bg-open-book.png", OUT / "cover-illustration-open-book.png", "card", BLUE),
    ]
    for source, target, mode, accent in candidates:
        if source.exists():
            cover_generated_background(source, target, mode=mode, accent=accent)


if __name__ == "__main__":
    cover_slate_grid()
    cover_phase_axis()
    cover_token_river()
    cover_minimal_axis()
    cover_ai_illustrations()
    cover_files = [
        OUT / "cover-illustration-token-core.png",
        OUT / "cover-illustration-model-engine.png",
        OUT / "cover-illustration-open-book.png",
        OUT / "cover-slate-grid.png",
        OUT / "cover-phase-axis.png",
        OUT / "cover-token-river.png",
        OUT / "cover-minimal-axis.png",
    ]
    cover_files = [p for p in cover_files if p.exists()]
    cols = 3
    rows = math.ceil(len(cover_files) / cols)
    cell_w, cell_h = 430, 645
    thumb_w, thumb_h = 380, 570
    sheet = Image.new("RGB", (cols * cell_w + 60, rows * cell_h + 40), (245, 247, 249))
    draw = ImageDraw.Draw(sheet)
    positions = []
    for idx in range(len(cover_files)):
        row = idx // cols
        col = idx % cols
        positions.append((30 + col * cell_w, 40 + row * cell_h))
    for idx, (cover_file, pos) in enumerate(zip(cover_files, positions), 1):
        thumb = Image.open(cover_file).resize((thumb_w, thumb_h))
        sheet.paste(thumb, pos)
        draw.text(
            (pos[0], pos[1] + thumb_h + 18),
            f"{idx}. {cover_file.stem}",
            font=font(FONT_BOLD, 22),
            fill=INK,
        )
    sheet.save(OUT / "cover-candidates-contact-sheet.png", quality=95)
    print(f"Wrote cover candidates to {OUT}")
