# -*- coding: utf-8 -*-
"""
✂️ QR Cutter — Telegram-бот для автоматической вырезки QR-кодов из фотографий.

Один файл, рассчитан на запуск в Pydroid 3 (работает и на ПК).
Секции файла соответствуют модулям проекта:
  1. Конфигурация       (config.py)
  2. Обработка QR       (services/qr_processor.py)
  3. База данных        (database/db.py)
  4. Интерфейс/тексты
  5. Обработчики        (handlers/start.py, photo.py, history.py)
  6. Запуск             (bot.py)

Зависимости: pyTelegramBotAPI, pillow, numpy, python-dotenv (OpenCV не нужен)
"""
from __future__ import annotations

import html
import io
import logging
import os
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- проверка зависимостей (Pydroid 3) ---
import importlib
import subprocess

# Библиотеки с C-кодом: на Android pip с обычного PyPI их НЕ ставит.
# Их нужно ставить через меню Pydroid (☰ → Pip) с плагином Pydroid repository plugin.
_HEAVY = (("numpy", "numpy"), ("PIL", "pillow"))
# Лёгкие чистые Python-пакеты: ставятся автоматически.
_LIGHT = (("dotenv", "python-dotenv"), ("telebot", "pyTelegramBotAPI"))


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def _pip_install(spec: str) -> bool:
    print(f"📦 Устанавливаю {spec} ...")
    try:
        if subprocess.run([sys.executable, "-m", "pip", "install", spec]).returncode == 0:
            return True
    except Exception:
        pass
    try:  # запасной путь для Pydroid
        from pip._internal.cli.main import main as _pip_main
        return _pip_main(["install", spec]) == 0
    except Exception as e:
        print("   не получилось:", e)
        return False


_miss_heavy = [spec for mod, spec in _HEAVY if not _have(mod)]
if _miss_heavy:
    print("❌ Не установлены библиотеки: " + ", ".join(_miss_heavy))
    print("""
Они нужны для обработки картинок. На Android их надо ставить через меню Pydroid:

1) Pydroid 3 → ☰ (меню) → Pip → вкладка «INSTALL».
2) Поставь галочку «Use prebuilt libraries repository».
3) Вводи по одному и жми INSTALL:
""" + "\n".join("      " + x for x in _miss_heavy) + """
4) Запусти этот файл снова.
(OpenCV для этого бота НЕ нужен.)
""")
    sys.exit(1)

for _mod, _spec in _LIGHT:
    if not _have(_mod):
        _pip_install(_spec)
        importlib.invalidate_caches()
        if not _have(_mod):
            print(f"\n❌ Не удалось установить {_spec}.")
            print("Поставь вручную: Pydroid 3 → ☰ → Pip → «Установка» → " + _spec)
            sys.exit(1)

import itertools

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps

import threading

import telebot
from telebot.apihelper import ApiTelegramException
from telebot.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)


# ═══════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent

# Токен берётся из переменной окружения BOT_TOKEN или из файла .env
# (если файл .env не виден в Pydroid — подойдёт и env.txt рядом со скриптом).
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "env.txt")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
QR_DIR = DATA_DIR / "qr"
DB_PATH = DATA_DIR / "history.db"
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "15"))        # макс. размер входящего файла
RATE_LIMIT_SEC = float(os.getenv("RATE_LIMIT_SEC", "3"))  # мин. пауза между изображениями
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))              # записей истории на странице
MAX_CODES = 20                                            # макс. QR-кодов за сессию / с фото
MAX_PIXELS = 60_000_000                                   # защита от «бомб» декомпрессии
WORK_SIDE = 1800                                          # макс. сторона для поиска QR

DATA_DIR.mkdir(parents=True, exist_ok=True)
QR_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("qr-cutter")


# ═══════════════════════════════════════════════════════════════════
# 2. ОБРАБОТКА QR-КОДОВ (только numpy + Pillow, без OpenCV)
# ═══════════════════════════════════════════════════════════════════
class InvalidImage(Exception):
    """Файл не является корректным изображением."""


@dataclass
class QRCrop:
    png: bytes
    content: str | None


def load_image(data: bytes) -> Image.Image:
    """Безопасно открывает изображение, исправляет поворот по EXIF, убирает прозрачность."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                return bg
            return im.convert("RGB")
    except Exception as e:  # повреждённый файл, не картинка, слишком большой и т.д.
        raise InvalidImage(str(e)) from e


# ---------- бинаризация ----------
def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    s0 = np.cumsum(hist * levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = s0 / w0
        m1 = (s0[-1] - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    return int(np.argmax(np.nan_to_num(between)))


def _binarize(gray: np.ndarray, method: str) -> np.ndarray:
    """Возвращает булеву маску «тёмный пиксель»."""
    if method == "otsu":
        return gray <= _otsu(gray)
    r = max(15, min(gray.shape) // 16)
    blur = np.asarray(Image.fromarray(gray).filter(ImageFilter.BoxBlur(r)), dtype=np.int16)
    return gray.astype(np.int16) < blur - 12


# ---------- поиск трёх «глаз» (finder patterns 1:1:3:1:1) ----------
def _scan_rows(dark: np.ndarray):
    """Строчный проход. Возвращает кластеры (x, y, unit, высота) центров глаз."""
    H, W = dark.shape
    step = max(1, H // 700)
    cands = []
    for y in range(0, H, step):
        row = dark[y]
        ch = np.flatnonzero(row[1:] != row[:-1]) + 1
        if len(ch) < 4:
            continue
        starts = np.concatenate(([0], ch))
        lens = (np.concatenate((ch, [W])) - starts).astype(np.float64)
        col = row[starts]
        l0, l1, l2, l3, l4 = lens[:-4], lens[1:-3], lens[2:-2], lens[3:-1], lens[4:]
        u = (l0 + l1 + l2 + l3 + l4) / 7.0
        tol = u * 0.65
        ok = (col[:-4] & (u >= 1.5)
              & (np.abs(l0 - u) <= tol) & (np.abs(l1 - u) <= tol)
              & (np.abs(l3 - u) <= tol) & (np.abs(l4 - u) <= tol)
              & (np.abs(l2 - 3 * u) <= u * 1.3))
        for i in np.flatnonzero(ok):
            cands.append((starts[i + 2] + lens[i + 2] / 2.0, float(y), float(u[i])))

    if len(cands) > 8000:  # слишком много «совпадений» — это шум/текстура
        cands = cands[:: len(cands) // 8000 + 1]
    done: list[dict] = []
    active: list[dict] = []
    for cx, y, u in cands:  # кандидаты идут по возрастанию y
        still = []
        for c in active:
            if y - c["y1"] > 2.5 * step:
                done.append(c)
            else:
                still.append(c)
        active = still
        for c in active:
            if abs(cx - c["cx"]) <= c["u"]:
                c["n"] += 1
                c["cx"] += (cx - c["cx"]) / c["n"]
                c["u"] += (u - c["u"]) / c["n"]
                c["y1"] = y
                break
        else:
            active.append({"cx": cx, "y0": y, "y1": y, "u": u, "n": 1})
    done += active
    good = [c for c in done if c["y1"] - c["y0"] + step >= 0.3 * c["u"]]
    good.sort(key=lambda c: -c["n"])
    return [(c["cx"], (c["y0"] + c["y1"]) / 2.0, c["u"]) for c in good[:250]]


def _run_mid(line: np.ndarray, pos: int):
    """Середина и длина тёмной серии, содержащей позицию pos."""
    if not line[pos]:
        return None
    n = len(line)
    lo = pos
    while lo > 0 and line[lo - 1]:
        lo -= 1
    hi = pos
    while hi < n - 1 and line[hi + 1]:
        hi += 1
    return (lo + hi + 1) / 2.0, hi - lo + 1


def _refine_center(dark: np.ndarray, x: float, y: float, u: float):
    """Уточняет центр «глаза» по границам центрального тёмного квадрата 3×3 модуля."""
    H, W = dark.shape
    for _ in range(2):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            return None
        h = _run_mid(dark[yi], xi)
        v = _run_mid(dark[:, xi], yi)
        if h is None or v is None:
            return None
        if not (1.8 * u <= h[1] <= 4.5 * u and 1.8 * u <= v[1] <= 4.5 * u):
            return None
        x, y = h[0], v[0]
        u = (h[1] + v[1]) / 6.0
    return x, y, u


def _finders(dark: np.ndarray) -> list[tuple[float, float, float]]:
    """Центры глаз, подтверждённые и по горизонтали, и по вертикали."""
    hor = _scan_rows(dark)
    ver = [(y, x, u) for x, y, u in _scan_rows(np.ascontiguousarray(dark.T))]  # (x, y, u)
    found: list[tuple[float, float, float]] = []
    for hx, hy, hu in hor:
        best = None
        for vx, vy, vu in ver:
            d = ((hx - vx) ** 2 + (hy - vy) ** 2) ** 0.5
            if d <= 1.5 * max(hu, vu) and (best is None or d < best[0]):
                best = (d, vx, vy, vu)
        if best is None:
            continue
        x, y, u = (hx + best[1]) / 2, (hy + best[2]) / 2, (hu + best[3]) / 2
        refined = _refine_center(dark, x, y, u)
        if refined is None:
            continue
        x, y, u = refined
        if any(((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 < 3 * u for fx, fy, _ in found):
            continue
        found.append((x, y, u))
    return found[:30]


def _group(finders):
    """Все правдоподобные тройки глаз, лучшие первыми: (score, ids, A, B, C, n_оценка)."""
    cands = []
    for i, j, k in itertools.combinations(range(len(finders)), 3):
        trio = [finders[i], finders[j], finders[k]]
        p = [np.array(t[:2]) for t in trio]
        us = [t[2] for t in trio]
        d = {(0, 1): np.linalg.norm(p[0] - p[1]), (0, 2): np.linalg.norm(p[0] - p[2]),
             (1, 2): np.linalg.norm(p[1] - p[2])}
        far = max(d, key=d.get)
        a = ({0, 1, 2} - set(far)).pop()
        b, c = sorted(set(far))
        ab, ac = d[tuple(sorted((a, b)))], d[tuple(sorted((a, c)))]
        if min(ab, ac) / max(ab, ac) < 0.7:
            continue
        va, vb = p[b] - p[a], p[c] - p[a]
        cosang = abs(float(np.dot(va, vb))) / (ab * ac)
        if cosang > 0.35 or max(us) / min(us) > 1.7:
            continue
        ms = sum(us) / 3
        s = (ab + ac) / 2 / ms  # = (N-7) для неповёрнутого кода; при повороте на 45° меньше в √2 раз
        if s < 8.5:
            continue
        ns = [n for n in range(21, 178, 4) if 0.93 * s <= n - 7 <= 1.5 * s]
        if not ns:
            continue
        score = (1 - min(ab, ac) / max(ab, ac)) + cosang + 0.5 * (max(us) / min(us) - 1)
        A, B, C = p[a], p[b], p[c]
        if (B - A)[0] * (C - A)[1] - (B - A)[1] * (C - A)[0] < 0:
            B, C = C, B
        cands.append((score, (i, j, k), A, B, C, ns))
    cands.sort(key=lambda t: t[0])
    return cands


_FINDER = np.zeros((7, 7), dtype=bool)
_FINDER[0, :] = _FINDER[6, :] = _FINDER[:, 0] = _FINDER[:, 6] = True
_FINDER[2:5, 2:5] = True


def _affine_h(A, B, C, n: int) -> np.ndarray:
    """Матрица перехода (столбец, строка модуля) → пиксели по трём центрам «глаз»."""
    mu, mv = (B - A) / (n - 7), (C - A) / (n - 7)
    o = A - 3 * mu - 3 * mv
    return np.array([[mu[0], mv[0], o[0]], [mu[1], mv[1], o[1]], [0.0, 0.0, 1.0]])


def _apply_h(Hm: np.ndarray, cols, rows):
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    w = Hm[2, 0] * cols + Hm[2, 1] * rows + Hm[2, 2]
    return ((Hm[0, 0] * cols + Hm[0, 1] * rows + Hm[0, 2]) / w,
            (Hm[1, 0] * cols + Hm[1, 1] * rows + Hm[1, 2]) / w)


def _hom_from_points(src, dst) -> np.ndarray | None:
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    try:
        h = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    return np.append(h, 1.0).reshape(3, 3)


_ALIGN = np.ones((5, 5), dtype=bool)
_ALIGN[1:4, 1:4] = False
_ALIGN[2, 2] = True


def _module_len(Hm: np.ndarray, n: int) -> float:
    x0, y0 = _apply_h(Hm, n / 2, n / 2)
    x1, y1 = _apply_h(Hm, n / 2 + 1, n / 2)
    return float(np.hypot(x1 - x0, y1 - y0))


def _window_means(ii: np.ndarray, X, Y, w: int):
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    xi, yi = np.rint(X).astype(int), np.rint(Y).astype(int)
    x0, x1 = np.clip(xi - w, 0, W - 1), np.clip(xi + w + 1, 1, W)
    y0, y1 = np.clip(yi - w, 0, H - 1), np.clip(yi + w + 1, 1, H)
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area


def _with_alignment(ii: np.ndarray, A, B, C, n: int) -> np.ndarray | None:
    """Находит выравнивающий узор у правого нижнего угла и строит перспективную матрицу."""
    if n < 25:
        return None
    base = _affine_h(A, B, C, n)
    L = _module_len(base, n)
    w = max(1, int(round(0.22 * L)))
    off = np.arange(-2.5, 2.51, 0.25)
    dc, dr = np.meshgrid(off, off)                       # смещения центра в модулях
    ij = np.arange(-2, 3)
    ic, ir = np.meshgrid(ij, ij)
    cols = (n - 7) + dc[..., None, None] + ic            # (21,21,5,5)
    rows = (n - 7) + dr[..., None, None] + ir
    X, Y = _apply_h(base, cols, rows)
    dark = _window_means(ii, X, Y, w) > 0.5
    score = (dark == _ALIGN).sum(axis=(2, 3))
    k = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[k] < 23:
        return None
    cx, cy = _apply_h(base, (n - 7) + dc[k], (n - 7) + dr[k])
    src = [(3, 3), (n - 4, 3), (3, n - 4), (n - 7, n - 7)]
    dst = [tuple(A), tuple(B), tuple(C), (float(cx), float(cy))]
    return _hom_from_points(src, dst)


def _corner_points(Hm: np.ndarray, n: int) -> np.ndarray:
    xs, ys = _apply_h(Hm, [-0.5, n - 0.5, -0.5, n - 0.5], [-0.5, -0.5, n - 0.5, n - 0.5])
    return np.stack([xs, ys], axis=1)


def _grid_quality(M: np.ndarray, n: int) -> tuple[float, float]:
    idx = np.arange(8, n - 8)
    t = ((M[6, 8:n - 8] == (idx % 2 == 0)).mean() + (M[8:n - 8, 6] == (idx % 2 == 0)).mean()) / 2
    f = ((M[:7, :7] == _FINDER).mean() + (M[:7, n - 7:] == _FINDER).mean()
         + (M[n - 7:, :7] == _FINDER).mean()) / 3
    return float(t), float(f)


def _fit_grid(masks, A, B, C, ns: list[int]):
    """Подбирает размер сетки и геометрию. Возвращает (качество, n, индекс маски, матрица) или None."""
    best = None
    for n in ns:
        for mi, ii in enumerate(masks):
            mats = [_affine_h(A, B, C, n)]
            persp = _with_alignment(ii, A, B, C, n)
            if persp is not None:
                mats.append(persp)
            for Hm in mats:
                t, f = _grid_quality(_sample(ii, Hm, n), n)
                if t < 0.78 or f < 0.78:
                    continue
                q = t + f
                if best is None or q > best[0]:
                    best = (q, n, mi, Hm)
    return best


# ---------- считывание содержимого QR (без внешних библиотек) ----------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _div(a: int, b: int) -> int:
    return 0 if a == 0 else _EXP[(_LOG[a] - _LOG[b]) % 255]


def _rs_fix(cw: list[int], nsym: int) -> list[int] | None:
    """Reed–Solomon: исправляет ошибки в блоке (данные + ECC)."""
    n = len(cw)
    S = []
    for i in range(nsym):
        s = 0
        for c in cw:
            s = _mul(s, _EXP[i]) ^ c
        S.append(s)
    if not any(S):
        return cw
    C, B, L, m, b = [1], [1], 0, 1, 1
    for k in range(nsym):
        d = S[k]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= _mul(C[i], S[k - i])
        if d == 0:
            m += 1
            continue
        coef = _div(d, b)
        T = C[:]
        if len(C) < len(B) + m:
            C = C + [0] * (len(B) + m - len(C))
        for i, bi in enumerate(B):
            C[i + m] ^= _mul(coef, bi)
        if 2 * L <= k:
            L, B, b, m = k + 1 - L, T, d, 1
        else:
            m += 1
    if L > nsym // 2:
        return None
    errs = []
    for k in range(n):
        xinv = _EXP[(255 - (n - 1 - k)) % 255]
        v = 0
        for coefficient in reversed(C):
            v = _mul(v, xinv) ^ coefficient
        if v == 0:
            errs.append(k)
    if len(errs) != L:
        return None
    omega = [0] * nsym
    for i in range(nsym):
        for j in range(len(C)):
            if i + j < nsym:
                omega[i + j] ^= _mul(S[i], C[j])
    out = cw[:]
    for k in errs:
        X = _EXP[(n - 1 - k) % 255]
        xinv = _EXP[(255 - (n - 1 - k)) % 255]
        num = 0
        for coefficient in reversed(omega):
            num = _mul(num, xinv) ^ coefficient
        den = 0
        for i in range(1, len(C), 2):
            den ^= _mul(C[i], _EXP[(_LOG[xinv] * (i - 1)) % 255] if xinv else 0)
        if den == 0:
            return None
        out[k] ^= _mul(X, _div(num, den))
    # проверка
    for i in range(nsym):
        s = 0
        for c in out:
            s = _mul(s, _EXP[i]) ^ c
        if s:
            return None
    return out


# версия: ((данные, блоков) для уровней L, M, Q, H)
_BLOCKS = {
    1: ((19, 1), (16, 1), (13, 1), (9, 1)), 2: ((34, 1), (28, 1), (22, 1), (16, 1)),
    3: ((55, 1), (44, 1), (34, 2), (26, 2)), 4: ((80, 1), (64, 2), (48, 2), (36, 4)),
    5: ((108, 1), (86, 2), (62, 4), (46, 4)), 6: ((136, 2), (108, 4), (76, 4), (60, 4)),
    7: ((156, 2), (124, 4), (88, 6), (66, 5)), 8: ((194, 2), (154, 4), (110, 6), (86, 6)),
    9: ((232, 2), (182, 5), (132, 8), (100, 8)), 10: ((274, 4), (216, 5), (154, 8), (122, 8)),
}
_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def _format_table() -> dict[int, int]:
    table = {}
    for data in range(32):
        rem = data
        for _ in range(10):
            rem = (rem << 1) ^ ((rem >> 9) * 0x537)
        table[((data << 10) | rem) ^ 0x5412] = data
    return table


_FORMATS = _format_table()


def _read_format(M: np.ndarray) -> tuple[int, int] | None:
    n = M.shape[0]
    g = lambda x, y: int(M[y, x])
    a = 0
    for i in range(0, 6):
        a |= g(8, i) << i
    a |= g(8, 7) << 6
    a |= g(8, 8) << 7
    a |= g(7, 8) << 8
    for i in range(9, 15):
        a |= g(14 - i, 8) << i
    b = 0
    for i in range(0, 8):
        b |= g(n - 1 - i, 8) << i
    for i in range(8, 15):
        b |= g(8, n - 15 + i) << i
    best = None
    for word in (a, b):
        for code, data in _FORMATS.items():
            dist = bin(word ^ code).count("1")
            if dist <= 3 and (best is None or dist < best[0]):
                best = (dist, data)
    if best is None:
        return None
    return best[1] >> 3, best[1] & 7  # (биты уровня ECC, маска)


def _func_mask(v: int) -> np.ndarray:
    n = 4 * v + 17
    F = np.zeros((n, n), dtype=bool)
    F[:9, :9] = True
    F[:9, n - 8:] = True
    F[n - 8:, :9] = True
    F[6, :] = True
    F[:, 6] = True
    if v >= 2:
        na = v // 7 + 2
        step = (v * 8 + na * 3 + 5) // (na * 4 - 4) * 2
        pos = sorted([6] + [v * 4 + 10 - i * step for i in range(na - 1)])
        for cy in pos:
            for cx in pos:
                if (cx == 6 and cy == 6) or (cx == 6 and cy == pos[-1]) or (cx == pos[-1] and cy == 6):
                    continue
                F[cy - 2:cy + 3, cx - 2:cx + 3] = True
    if v >= 7:
        F[:6, n - 11:n - 8] = True
        F[n - 11:n - 8, :6] = True
    return F


_MASKS = (
    lambda x, y: (x + y) % 2 == 0,
    lambda x, y: y % 2 == 0,
    lambda x, y: x % 3 == 0,
    lambda x, y: (x + y) % 3 == 0,
    lambda x, y: (x // 3 + y // 2) % 2 == 0,
    lambda x, y: x * y % 2 + x * y % 3 == 0,
    lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
    lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
)


def _total_codewords(v: int) -> int:
    r = (16 * v + 128) * v + 64
    if v >= 2:
        na = v // 7 + 2
        r -= (25 * na - 10) * na - 55
        if v >= 7:
            r -= 36
    return r // 8


def _decode_matrix(M: np.ndarray) -> str | None:
    n = M.shape[0]
    if (n - 17) % 4 or not 21 <= n <= 57:
        return None
    v = (n - 17) // 4
    fmt = _read_format(M)
    if fmt is None:
        return None
    ecl_bits, mask_id = fmt
    level = {1: 0, 0: 1, 3: 2, 2: 3}[ecl_bits]
    F = _func_mask(v)
    total = _total_codewords(v)
    maskf = _MASKS[mask_id]
    bits: list[int] = []
    for right in range(n - 1, 0, -2):
        if right <= 6:
            right -= 1
        upward = ((right + 1) & 2) == 0
        for vert in range(n):
            y = n - 1 - vert if upward else vert
            for j in (0, 1):
                x = right - j
                if not F[y, x]:
                    bits.append(int(M[y, x]) ^ int(maskf(x, y)))
    if len(bits) < total * 8:
        return None
    cw = [int("".join(map(str, bits[i * 8:i * 8 + 8])), 2) for i in range(total)]

    data_cw, nb = _BLOCKS[v][level]
    ecc = (total - data_cw) // nb
    n_short = nb - total % nb
    short_data = total // nb - ecc
    lens = [short_data + (0 if j < n_short else 1) for j in range(nb)]
    blocks = [[] for _ in range(nb)]
    pos = 0
    for i in range(short_data + 1):
        for j in range(nb):
            if i < lens[j]:
                blocks[j].append(cw[pos])
                pos += 1
    eccs = [[] for _ in range(nb)]
    for i in range(ecc):
        for j in range(nb):
            eccs[j].append(cw[pos])
            pos += 1
    data: list[int] = []
    for j in range(nb):
        fixed = _rs_fix(blocks[j] + eccs[j], ecc)
        if fixed is None:
            return None
        data += fixed[:lens[j]]
    return _parse_segments("".join(f"{b:08b}" for b in data), v)


def _parse_segments(bs: str, v: int) -> str | None:
    g = 0 if v <= 9 else 1 if v <= 26 else 2
    pos, out = 0, []
    while len(bs) - pos >= 4:
        mode = int(bs[pos:pos + 4], 2)
        pos += 4
        if mode == 0:
            break
        if mode == 1:
            cc = (10, 12, 14)[g]
            cnt = int(bs[pos:pos + cc], 2)
            pos += cc
            s = ""
            while cnt > 0:
                k = min(3, cnt)
                width = {3: 10, 2: 7, 1: 4}[k]
                s += str(int(bs[pos:pos + width], 2)).zfill(k)
                pos += width
                cnt -= k
            out.append(s)
        elif mode == 2:
            cc = (9, 11, 13)[g]
            cnt = int(bs[pos:pos + cc], 2)
            pos += cc
            s = ""
            while cnt > 0:
                if cnt >= 2:
                    val = int(bs[pos:pos + 11], 2)
                    pos += 11
                    s += _ALNUM[val // 45] + _ALNUM[val % 45]
                    cnt -= 2
                else:
                    s += _ALNUM[int(bs[pos:pos + 6], 2)]
                    pos += 6
                    cnt -= 1
            out.append(s)
        elif mode == 4:
            cc = (8, 16, 16)[g]
            cnt = int(bs[pos:pos + cc], 2)
            pos += cc
            raw = bytes(int(bs[pos + 8 * i:pos + 8 * i + 8], 2) for i in range(cnt))
            pos += 8 * cnt
            try:
                out.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(raw.decode("latin-1"))
        elif mode == 8:
            cc = (8, 10, 12)[g]
            cnt = int(bs[pos:pos + cc], 2)
            pos += cc
            raw = bytearray()
            for _ in range(cnt):
                val = int(bs[pos:pos + 13], 2)
                pos += 13
                val = (val // 0xC0 << 8) | (val % 0xC0)
                val += 0x8140 if val < 0x1F00 else 0xC140
                raw += bytes([val >> 8, val & 0xFF])
            out.append(raw.decode("shift_jis", errors="replace"))
        elif mode == 7:  # ECI — пропускаем обозначение кодировки
            first = bs[pos]
            pos += 8 if first == "0" else 16 if bs[pos:pos + 2] == "10" else 24
        else:
            return None
        if pos > len(bs):
            return None
    text = "".join(out)
    return text or None


def _sample(ii: np.ndarray, Hm: np.ndarray, n: int) -> np.ndarray:
    """Снимает матрицу модулей n×n по матрице геометрии."""
    rr, cc = np.mgrid[0:n, 0:n]
    X, Y = _apply_h(Hm, cc, rr)
    w = max(1, int(round(0.22 * _module_len(Hm, n))))
    return _window_means(ii, X, Y, w) > 0.5


def _read_content(masks, Hm: np.ndarray, A, B, C, n: int, first_mask: int) -> str | None:
    """Пробует считать текст QR в сетке n×n."""
    if not 21 <= n <= 57:
        return None
    order = [first_mask] + [i for i in range(len(masks)) if i != first_mask]
    geoms = [Hm, _affine_h(A, B, C, n)]
    for mi in order:
        for G in geoms:
            M = _sample(masks[mi], G, n)
            for cand in (M, M.T):
                try:
                    text = _decode_matrix(cand)
                except Exception:
                    text = None
                if text:
                    return text
    return None


def _integral(dark: np.ndarray) -> np.ndarray:
    ii = np.zeros((dark.shape[0] + 1, dark.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = dark.astype(np.float64).cumsum(0).cumsum(1)
    return ii


def process_image(data: bytes) -> list[QRCrop]:
    """Главная функция: байты изображения → список вырезанных QR-кодов (PNG + текст)."""
    img = load_image(data)
    W0, H0 = img.size
    base = min(1.0, WORK_SIDE / max(W0, H0))
    work = img.resize((max(1, int(W0 * base)), max(1, int(H0 * base))), Image.LANCZOS) if base < 1 else img
    gray0 = np.asarray(work.convert("L"), dtype=np.uint8)

    attempts = [(1.0, "otsu", False), (1.0, "local", False), (0.5, "otsu", False),
                (0.5, "local", False), (1.0, "otsu", True), (1.0, "local", True)]
    found = []   # (A, B, C, n, content, scale)
    deadline = time.monotonic() + 25
    for scale, method, inv in attempts:
        if time.monotonic() > deadline:
            break
        if scale == 1.0:
            gray = gray0
        else:
            gray = np.asarray(
                Image.fromarray(gray0).resize(
                    (max(1, int(gray0.shape[1] * scale)), max(1, int(gray0.shape[0] * scale))),
                    Image.BOX), dtype=np.uint8)
        if min(gray.shape) < 40:
            continue
        dark = _binarize(gray, method)
        if inv:
            dark = ~dark
        cands = _group(_finders(dark))
        if not cands:
            continue
        other = _binarize(gray, "local" if method == "otsu" else "otsu")
        masks = [_integral(dark), _integral(~other if inv else other)]
        used: set[int] = set()
        for _score, ids, A, B, C, ns in cands[:80]:
            if used & set(ids):
                continue
            fit = _fit_grid(masks, A, B, C, ns)
            if fit is None:
                continue
            _q, n, mi, Hm = fit
            used |= set(ids)
            text = _read_content(masks, Hm, A, B, C, n, mi)
            found.append((Hm, n, text, scale))
        if found:
            break

    results: list[QRCrop] = []
    rects = []
    for Hm, n, text, scale in found:
        k = 1.0 / (base * scale)
        pts = _corner_points(Hm, n) * k
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        rects.append((x0, y0, x1, y1, text))
    rects.sort(key=lambda r: (round(r[1] / 50), r[0]))
    for x0, y0, x1, y1, text in rects[:MAX_CODES]:
        side = max(x1 - x0, y1 - y0)
        pad = max(int(side * 0.08), 8)
        box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
               min(W0, int(x1 + pad)), min(H0, int(y1 + pad)))
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            continue
        crop = img.crop(box)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        results.append(QRCrop(png=buf.getvalue(), content=text))
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. БАЗА ДАННЫХ (SQLite)
# ═══════════════════════════════════════════════════════════════════
class Database:
    def __init__(self, path: Path):
        self.path = path
        with closing(self._conn()) as c, c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS cuts (
                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id    INTEGER NOT NULL,
                       file_name  TEXT    NOT NULL,
                       content    TEXT,
                       created_at TEXT    NOT NULL
                   )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_cuts_user ON cuts(user_id, id DESC)")
            c.execute(
                """CREATE TABLE IF NOT EXISTS users (
                       user_id       INTEGER PRIMARY KEY,
                       first_name    TEXT    NOT NULL DEFAULT '',
                       registered_at TEXT    NOT NULL,
                       photos_count  INTEGER NOT NULL DEFAULT 0,
                       cuts_count    INTEGER NOT NULL DEFAULT 0
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, user_id: int, file_name: str, content: str | None, created_at: str) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO cuts (user_id, file_name, content, created_at) VALUES (?,?,?,?)",
                (user_id, file_name, content, created_at),
            )
            return int(cur.lastrowid)

    def count(self, user_id: int) -> int:
        with closing(self._conn()) as c:
            return c.execute("SELECT COUNT(*) FROM cuts WHERE user_id=?", (user_id,)).fetchone()[0]

    def page(self, user_id: int, limit: int, offset: int) -> list[sqlite3.Row]:
        with closing(self._conn()) as c:
            return c.execute(
                "SELECT * FROM cuts WHERE user_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()

    def get(self, user_id: int, cut_id: int) -> sqlite3.Row | None:
        # Условие user_id гарантирует, что чужие записи недоступны
        with closing(self._conn()) as c:
            return c.execute(
                "SELECT * FROM cuts WHERE id=? AND user_id=?", (cut_id, user_id)
            ).fetchone()

    def touch_user(self, user_id: int, first_name: str | None) -> None:
        """Создаёт профиль при первом обращении и обновляет имя."""
        name = (first_name or "").strip()
        with closing(self._conn()) as c, c:
            row = c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,)).fetchone()
            if row is None:
                old = c.execute(
                    "SELECT MIN(created_at) AS d, COUNT(*) AS n FROM cuts WHERE user_id=?", (user_id,)
                ).fetchone()
                registered = old["d"] or datetime.now().isoformat(timespec="seconds")
                n = int(old["n"] or 0)
                c.execute(
                    "INSERT INTO users (user_id, first_name, registered_at, photos_count, cuts_count)"
                    " VALUES (?,?,?,?,?)",
                    (user_id, name, registered, n, n),
                )
            elif name:
                c.execute("UPDATE users SET first_name=? WHERE user_id=?", (name, user_id))

    def bump(self, user_id: int, photos: int, cuts: int) -> None:
        with closing(self._conn()) as c, c:
            c.execute(
                "UPDATE users SET photos_count=photos_count+?, cuts_count=cuts_count+? WHERE user_id=?",
                (photos, cuts, user_id),
            )

    def user(self, user_id: int) -> sqlite3.Row | None:
        with closing(self._conn()) as c:
            return c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    def clear(self, user_id: int) -> list[str]:
        with closing(self._conn()) as c, c:
            names = [r["file_name"] for r in c.execute(
                "SELECT file_name FROM cuts WHERE user_id=?", (user_id,))]
            c.execute("DELETE FROM cuts WHERE user_id=?", (user_id,))
            return names


db = Database(DB_PATH)


# ═══════════════════════════════════════════════════════════════════
# 4. ИНТЕРФЕЙС: ТЕКСТЫ И КЛАВИАТУРЫ
# ═══════════════════════════════════════════════════════════════════
LINE = "━━━━━━━━━━━━━━━━"

BTN_HOW = "✂️ Как пользоваться"      # старые кнопки: оставлены только как обработчики
BTN_HISTORY = "📚 История нарезок"
BTN_ABOUT = "ℹ️ О боте"
BTN_PROFILE = "Профиль"
BTN_START = "Начать"
BTN_CANCEL = "Отмена"


def em(emoji_id: str, fallback: str) -> str:
    """Премиум-эмодзи (если недоступно — Telegram покажет обычный)."""
    return f'<tg-emoji emoji-id="{emoji_id}">{fallback}</tg-emoji>'


E_PLANE = em("5258073068852485953", "✈️")
E_INFO = em("5258503720928288433", "ℹ️")
E_IMG = em("5257974976094412956", "🖼")
E_USER = em("5316727448644103237", "👤")
E_CLOCK = em("5985616167740379273", "⏰")
E_DOWN = em("5258336354642697821", "⬇️")
E_UP = em("5258043150110301407", "⬆️")
E_OK = em("5260726538302660868", "✅")
E_DATE = em("5274055917766202507", "🗓")
E_NUM = em("5258330865674494479", "🍑")
E_LINK = em("5260730055880876557", "⛓")

WELCOME = (
    f"{E_PLANE} Привет! Добро пожаловать в QR Cutter!\n\n"
    f"{E_INFO} Я умею находить QR-коды на фотографиях и автоматически вырезать их за пару секунд.\n\n"
    f"{E_IMG} Нажми «Начать» или просто отправь фотографии — остальное я сделаю сам!"
)

START_PROMPT = (
    f"{E_IMG} <b>Режим загрузки QR-кодов</b>\n"
    "<blockquote>Отправь фотографии с QR-кодами (можно загрузить <b>до 20 QR</b> за раз — по одному или альбомом).\n"
    f"{E_INFO} Чтобы выйти из режима загрузки, нажми «Отмена» снизу.</blockquote>"
)

CANCEL_TEXT = (
    f"{E_INFO} <b>Загрузка отменена</b>\n"
    "<blockquote>Вы вернулись в главное меню.</blockquote>"
)

HELP = (
    "✂️ <b>Как пользоваться</b>\n"
    f"{LINE}\n"
    "1️⃣ Отправь фото с QR-кодом (как фото или как файл)\n"
    "2️⃣ Подожди пару секунд ⏳\n"
    "3️⃣ Получи аккуратно вырезанный QR-код\n\n"
    "📌 <b>Полезно знать</b>\n"
    "• Если на фото несколько QR-кодов — пришлю каждый отдельно\n"
    "• Кнопка «📁 Без сжатия» отдаёт файл в исходном качестве\n"
    f"• Все результаты хранятся в «{BTN_HISTORY}»\n"
    f"• Размер файла — до {MAX_FILE_MB} МБ\n\n"
    "⌨️ <b>Команды</b>\n"
    "/start — главное меню\n"
    "/help — эта инструкция\n"
    "/history — история нарезок\n"
    "/clear_history — очистить историю"
)

ABOUT = (
    "ℹ️ <b>О боте</b>\n"
    f"{LINE}\n"
    "✂️ <b>QR Cutter</b> — находит QR-код на фото и вырезает его по границам.\n\n"
    "🛡 Ссылки из QR-кодов я <b>никогда не открываю</b> — только показываю текст.\n"
    "🔒 История личная: вижу и отдаю только твои записи."
)

EMPTY_HISTORY = (
    "📭 <b>Здесь пока пусто!</b>\n\n"
    "Отправь свою первую фотографию, и обработанные QR-коды появятся здесь ✨"
)

NOT_FOUND = (
    "🔍 <b>QR-код не найден</b>\n\n"
    "Не получилось найти QR-код на этом снимке 😔\n"
    "Попробуй сфотографировать ближе, при хорошем освещении и без бликов — "
    "и отправь другое изображение 📸"
)


def btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def inline(*rows: list) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    for row in rows:
        kb.row(*row)
    return kb


def _kb_btn(text: str, emoji_id: str) -> KeyboardButton:
    try:
        return KeyboardButton(text, icon_custom_emoji_id=emoji_id)
    except TypeError:
        b = KeyboardButton(text)
        b.icon_custom_emoji_id = emoji_id
        return b


_orig_kb_to_dict = KeyboardButton.to_dict


def _patched_kb_to_dict(self):
    d = _orig_kb_to_dict(self)
    if getattr(self, "icon_custom_emoji_id", None):
        d["icon_custom_emoji_id"] = self.icon_custom_emoji_id
    return d


KeyboardButton.to_dict = _patched_kb_to_dict


def main_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="📸 Отправь фото с QR-кодом")
    kb.row(
        _kb_btn(BTN_PROFILE, "5258330865674494479"),
        _kb_btn(BTN_START, "5257974976094412956"),
    )
    return kb


def cancel_kb() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True, input_field_placeholder="📸 Отправь до 20 QR-кодов или нажми Отмена")
    kb.row(_kb_btn(BTN_CANCEL, "5258503720928288433"))
    return kb


def result_kb(cut_id: int) -> InlineKeyboardMarkup:
    return inline(
        [btn("📚 История нарезок", "h:0"), btn("🔄 Обработать ещё", "m:again")],
        [btn("📁 Без сжатия", f"f:{cut_id}")],
    )


def fmt_dt(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M:%S")
    except ValueError:
        return iso


def short(text: str | None, n: int) -> str:
    if not text:
        return "—"
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def caption_for(cut_id: int, created_at: str, content: str | None, success: bool) -> str:
    head = f"{E_OK} QR-код успешно обработан!" if success else f"{E_IMG} QR-код #{cut_id}"
    body = (
        f"{head}\n\n"
        f"{E_DATE} Дата: {fmt_dt(created_at)}\n"
        f"{E_NUM} Номер операции: {cut_id}\n"
    )
    if content:
        body += f"{E_LINK} Содержимое: {html.escape(short(content, 600))}"
    else:
        body += f"{E_LINK} Содержимое: <i>не удалось считать</i>"
    return body


# ═══════════════════════════════════════════════════════════════════
# 5. ОБРАБОТЧИКИ (pyTelegramBotAPI)
# ═══════════════════════════════════════════════════════════════════
class _ExHandler(telebot.ExceptionHandler):
    def handle(self, exception):
        log.error("Ошибка в обработчике: %s", exception, exc_info=exception)
        return True  # не даём боту упасть


if not BOT_TOKEN or ":" not in BOT_TOKEN:
    print("❌ Не найден BOT_TOKEN.\n"
          "Создайте файл .env (или env.txt) рядом со скриптом со строкой:\n"
          "BOT_TOKEN=123456:ABC...\n"
          "(или переменную окружения BOT_TOKEN).")
    sys.exit(1)

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML", threaded=True, num_threads=4,
                      exception_handler=_ExHandler())

_state_lock = threading.Lock()
_user_locks: dict[int, threading.Lock] = {}
_session_counts: dict[int, int] = {}
_session_active: set[int] = set()
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def tg(fn, *args, **kwargs):
    """Вызов Telegram API с повтором при флуд-лимите (429)."""
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except ApiTelegramException as e:
            if e.error_code == 429 and attempt < 2:
                try:
                    delay = int(e.result_json["parameters"]["retry_after"])
                except Exception:
                    delay = 3
                time.sleep(delay + 0.5)
                continue
            raise


def named(data: bytes, name: str) -> io.BytesIO:
    f = io.BytesIO(data)
    f.name = name
    return f


def send_image(chat_id: int, data: bytes, name: str, caption: str, kb,
               reply_to_message_id: int | None = None) -> None:
    """Отправляет картинку фото (ответом на сообщение); если Telegram не принял — файлом."""
    try:
        tg(bot.send_photo, chat_id, named(data, name), caption=caption, reply_markup=kb,
           reply_to_message_id=reply_to_message_id, allow_sending_without_reply=True)
    except ApiTelegramException as e:
        if e.error_code == 429:
            raise
        tg(bot.send_document, chat_id, named(data, name), caption=caption, reply_markup=kb,
           reply_to_message_id=reply_to_message_id, allow_sending_without_reply=True)


def edit_or_send(cb: CallbackQuery, text: str, kb) -> None:
    """Редактирует сообщение; если нельзя (например, это фото) — отправляет новое."""
    msg = cb.message
    try:
        bot.edit_message_text(text, msg.chat.id, msg.message_id, reply_markup=kb)
    except ApiTelegramException as e:
        if "not modified" in str(e).lower():
            return
        tg(bot.send_message, msg.chat.id, text, reply_markup=kb)


# ---------- старт / помощь / о боте ----------
@bot.message_handler(commands=["start"])
def cmd_start(m: Message):
    uid = m.from_user.id if m.from_user else m.chat.id
    with _state_lock:
        _session_active.discard(uid)
        _session_counts[uid] = 0
    db.touch_user(uid, m.from_user.first_name if m.from_user else None)
    bot.send_message(m.chat.id, WELCOME, reply_markup=main_kb())


def _profile_text(user_id: int, first_name: str | None) -> str:
    db.touch_user(user_id, first_name)
    u = db.user(user_id)
    try:
        reg = datetime.fromisoformat(u["registered_at"]).strftime("%d.%m.%Y")
    except ValueError:
        reg = u["registered_at"]
    name = html.escape(u["first_name"] or "—")
    return (
        f"{E_USER} Имя: {name}\n"
        f"{E_INFO} Ваш ID: {user_id}\n"
        f"{E_CLOCK} Регистрация: {reg}\n\n"
        f"{E_DOWN} Количество QR: {u['photos_count']}\n"
        f"{E_UP} Количество обрезанных: {u['cuts_count']}"
    )


@bot.message_handler(commands=["profile"])
def cmd_profile(m: Message):
    bot.send_message(m.chat.id, _profile_text(m.from_user.id, m.from_user.first_name),
                     reply_markup=main_kb())


@bot.message_handler(func=lambda m: m.text in (BTN_PROFILE, "🍑 Профиль", "👤 Профиль", "Профиль"), content_types=["text"])
def btn_profile(m: Message):
    cmd_profile(m)


@bot.message_handler(func=lambda m: m.text in (BTN_START, "🖼 Начать", "Начать"), content_types=["text"])
def btn_start(m: Message):
    uid = m.from_user.id if m.from_user else m.chat.id
    with _state_lock:
        _session_active.add(uid)
        _session_counts[uid] = 0
    db.touch_user(uid, m.from_user.first_name if m.from_user else None)
    bot.send_message(m.chat.id, START_PROMPT, reply_markup=cancel_kb())


@bot.message_handler(func=lambda m: m.text in (BTN_CANCEL, "❌ Отмена", "Отмена"), content_types=["text"])
def btn_cancel(m: Message):
    uid = m.from_user.id if m.from_user else m.chat.id
    with _state_lock:
        _session_active.discard(uid)
        _session_counts[uid] = 0
    db.touch_user(uid, m.from_user.first_name if m.from_user else None)
    bot.send_message(m.chat.id, CANCEL_TEXT, reply_markup=main_kb())


@bot.message_handler(commands=["help"])
def cmd_help(m: Message):
    bot.send_message(m.chat.id, HELP, reply_markup=main_kb())


@bot.message_handler(func=lambda m: m.text == BTN_HOW, content_types=["text"])
def btn_help(m: Message):
    cmd_help(m)


@bot.message_handler(func=lambda m: m.text == BTN_ABOUT, content_types=["text"])
def about(m: Message):
    bot.send_message(m.chat.id, ABOUT, reply_markup=main_kb())


@bot.callback_query_handler(func=lambda c: c.data == "m:home")
def cb_home(cb: CallbackQuery):
    bot.send_message(cb.message.chat.id, WELCOME, reply_markup=main_kb())
    bot.answer_callback_query(cb.id)


@bot.callback_query_handler(func=lambda c: c.data == "m:again")
def cb_again(cb: CallbackQuery):
    bot.send_message(cb.message.chat.id, "📸 Жду следующее фото с QR-кодом!")
    bot.answer_callback_query(cb.id)


# ---------- обработка изображений ----------
def _get_user_lock(uid: int) -> threading.Lock:
    with _state_lock:
        lock = _user_locks.get(uid)
        if lock is None:
            lock = threading.Lock()
            _user_locks[uid] = lock
        return lock


def handle_image(m: Message, file_id: str, size: int | None) -> None:
    chat_id = m.chat.id
    uid = m.from_user.id if m.from_user else chat_id
    db.touch_user(uid, m.from_user.first_name if m.from_user else None)

    with _get_user_lock(uid):
        with _state_lock:
            in_session = uid in _session_active
            current_count = _session_counts.get(uid, 0)
            if in_session and current_count >= MAX_CODES:
                _session_active.discard(uid)
                _session_counts[uid] = 0
                limit_reached = True
            else:
                limit_reached = False

        if limit_reached:
            bot.send_message(
                chat_id,
                f"{E_INFO} <b>Лимит в {MAX_CODES} QR-кодов достигнут.</b>\n"
                "<blockquote>Нажми «Начать», чтобы загрузить новую партию.</blockquote>",
                reply_markup=main_kb(),
            )
            return

        kb = cancel_kb() if in_session else main_kb()

        try:
            if size and size > MAX_FILE_MB * 1024 * 1024:
                bot.send_message(chat_id, f"📦 Файл слишком большой. Максимум — {MAX_FILE_MB} МБ.", reply_markup=kb)
                return

            try:
                info = tg(bot.get_file, file_id)
                data = tg(bot.download_file, info.file_path)
            except ApiTelegramException:
                bot.send_message(chat_id, "😕 Не удалось скачать файл. Возможно, он слишком большой.", reply_markup=kb)
                return
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                bot.send_message(chat_id, f"📦 Файл слишком большой. Максимум — {MAX_FILE_MB} МБ.", reply_markup=kb)
                return

            try:
                crops = process_image(data)
            except InvalidImage:
                bot.send_message(
                    chat_id,
                    "🖼 <b>Не удалось открыть файл</b>\n\n"
                    "Похоже, изображение повреждено или это не картинка. "
                    "Отправь другое фото или файл в формате JPG / PNG / WEBP.",
                    reply_markup=kb,
                )
                return
            except Exception:
                log.exception("Ошибка обработки изображения")
                bot.send_message(chat_id, "⚠️ Что-то пошло не так при обработке. Попробуй другое фото.", reply_markup=kb)
                return
            finally:
                del data

            if not crops:
                bot.send_message(chat_id, NOT_FOUND, reply_markup=kb,
                                 reply_to_message_id=m.message_id, allow_sending_without_reply=True)
                return

            with _state_lock:
                in_session = uid in _session_active
                current_count = _session_counts.get(uid, 0)
                if in_session:
                    remaining = max(0, MAX_CODES - current_count)
                    crops = crops[:remaining]
                    _session_counts[uid] = current_count + len(crops)
                    reached_max = _session_counts[uid] >= MAX_CODES
                    if reached_max:
                        _session_active.discard(uid)
                        _session_counts[uid] = 0
                else:
                    reached_max = False

            if not crops:
                bot.send_message(
                    chat_id,
                    f"{E_INFO} <b>Лимит в {MAX_CODES} QR-кодов достигнут.</b>\n"
                    "<blockquote>Нажми «Начать», чтобы загрузить новую партию.</blockquote>",
                    reply_markup=main_kb(),
                )
                return

            db.bump(uid, 1, len(crops))
            out_kb = main_kb() if reached_max else (cancel_kb() if in_session else None)

            for i, crop in enumerate(crops, 1):
                now = datetime.now().isoformat(timespec="seconds")
                file_name = f"{uid}_{int(time.time() * 1000)}_{i}.png"
                (QR_DIR / file_name).write_bytes(crop.png)
                cut_id = db.add(uid, file_name, crop.content, now)
                caption = caption_for(cut_id, now, crop.content, success=True)
                send_image(
                    chat_id,
                    crop.png,
                    f"qr_{cut_id}.png",
                    caption,
                    out_kb if i == len(crops) else None,
                    reply_to_message_id=m.message_id,
                )

            if reached_max:
                bot.send_message(
                    chat_id,
                    f"{E_OK} <b>QR обработаны ({MAX_CODES}/{MAX_CODES})</b>\n"
                    "<blockquote>Все QR-коды успешно вырезаны и готовы!</blockquote>",
                    reply_markup=main_kb(),
                )
        except ApiTelegramException:
            log.exception("Ошибка Telegram API")
            try:
                bot.send_message(chat_id, "⚠️ Telegram временно не отвечает. Попробуй ещё раз чуть позже.")
            except ApiTelegramException:
                pass


@bot.message_handler(content_types=["photo"])
def on_photo(m: Message):
    p = m.photo[-1]
    handle_image(m, p.file_id, p.file_size)


@bot.message_handler(content_types=["document"])
def on_document(m: Message):
    d = m.document
    ext = Path(d.file_name or "").suffix.lower()
    if not ((d.mime_type or "").startswith("image/") or ext in ALLOWED_EXT):
        bot.send_message(m.chat.id, "📎 Этот формат не поддерживается.\n"
                                    "Отправь изображение: JPG, PNG, WEBP, BMP.")
        return
    handle_image(m, d.file_id, d.file_size)


# ---------- история ----------
def build_history(uid: int, page: int):
    total = db.count(uid)
    if total == 0:
        return EMPTY_HISTORY, inline([btn("🏠 Главное меню", "m:home")])

    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, pages - 1))
    rows = db.page(uid, PAGE_SIZE, page * PAGE_SIZE)

    lines = ["📚 <b>История нарезок</b>", LINE,
             f"Всего: <b>{total}</b> • Страница <b>{page + 1}/{pages}</b>\n"]
    item_buttons = []
    for r in rows:
        lines.append(
            f"🆔 <b>#{r['id']}</b> • 📅 {fmt_dt(r['created_at'])}\n"
            f"🔗 <code>{html.escape(short(r['content'], 45))}</code>\n"
        )
        item_buttons.append(btn(f"🖼 #{r['id']}", f"v:{r['id']}:{page}"))

    keyboard = [item_buttons[i:i + 3] for i in range(0, len(item_buttons), 3)]
    nav = []
    if page > 0:
        nav.append(btn("◀️ Назад", f"h:{page - 1}"))
    if page < pages - 1:
        nav.append(btn("Вперёд ▶️", f"h:{page + 1}"))
    if nav:
        keyboard.append(nav)
    keyboard.append([btn("🏠 Главное меню", "m:home")])
    lines.append("👇 Выбери запись, чтобы получить QR-код снова")
    return "\n".join(lines), inline(*keyboard)


@bot.message_handler(commands=["history"])
def cmd_history(m: Message):
    text, kb = build_history(m.from_user.id, 0)
    bot.send_message(m.chat.id, text, reply_markup=kb)


@bot.message_handler(func=lambda m: m.text == BTN_HISTORY, content_types=["text"])
def btn_history(m: Message):
    cmd_history(m)


@bot.callback_query_handler(func=lambda c: c.data.startswith("h:"))
def cb_history(cb: CallbackQuery):
    try:
        page = int(cb.data.split(":")[1])
    except (IndexError, ValueError):
        bot.answer_callback_query(cb.id)
        return
    text, kb = build_history(cb.from_user.id, page)
    edit_or_send(cb, text, kb)
    bot.answer_callback_query(cb.id)


@bot.callback_query_handler(func=lambda c: c.data.startswith("v:"))
def cb_view(cb: CallbackQuery):
    try:
        _, cid, page = cb.data.split(":")
        cut_id, page = int(cid), int(page)
    except ValueError:
        bot.answer_callback_query(cb.id)
        return
    row = db.get(cb.from_user.id, cut_id)
    path = QR_DIR / row["file_name"] if row else None
    if not row or not path.exists():
        bot.answer_callback_query(cb.id, "😕 Запись или файл не найдены", show_alert=True)
        return
    kb = inline(
        [btn("📁 Без сжатия", f"f:{cut_id}")],
        [btn("◀️ К истории", f"h:{page}"), btn("🏠 Главное меню", "m:home")],
    )
    caption = caption_for(cut_id, row["created_at"], row["content"], success=False)
    send_image(cb.message.chat.id, path.read_bytes(), f"qr_{cut_id}.png", caption, kb)
    bot.answer_callback_query(cb.id)


@bot.callback_query_handler(func=lambda c: c.data.startswith("f:"))
def cb_file(cb: CallbackQuery):
    try:
        cut_id = int(cb.data.split(":")[1])
    except (IndexError, ValueError):
        bot.answer_callback_query(cb.id)
        return
    row = db.get(cb.from_user.id, cut_id)
    path = QR_DIR / row["file_name"] if row else None
    if not row or not path.exists():
        bot.answer_callback_query(cb.id, "😕 Файл не найден", show_alert=True)
        return
    tg(bot.send_document, cb.message.chat.id, named(path.read_bytes(), f"qr_{cut_id}.png"),
       caption=f"📁 QR-код #{cut_id} в исходном качестве")
    bot.answer_callback_query(cb.id)


# ---------- очистка истории ----------
@bot.message_handler(commands=["clear_history"])
def cmd_clear(m: Message):
    total = db.count(m.from_user.id)
    if total == 0:
        bot.send_message(m.chat.id, EMPTY_HISTORY)
        return
    kb = inline([btn("✅ Да, удалить", "c:yes"), btn("❌ Отмена", "c:no")])
    bot.send_message(
        m.chat.id,
        f"🗑 <b>Очистка истории</b>\n{LINE}\n"
        f"Будет удалено записей: <b>{total}</b>.\nЭто действие нельзя отменить. Продолжить?",
        reply_markup=kb,
    )


@bot.callback_query_handler(func=lambda c: c.data == "c:yes")
def cb_clear_yes(cb: CallbackQuery):
    for n in db.clear(cb.from_user.id):
        try:
            (QR_DIR / n).unlink(missing_ok=True)
        except OSError:
            log.warning("Не удалось удалить файл %s", n)
    edit_or_send(cb, "🧹 <b>История очищена!</b>\nМожно начинать заново ✨", None)
    bot.answer_callback_query(cb.id, "Готово")


@bot.callback_query_handler(func=lambda c: c.data == "c:no")
def cb_clear_no(cb: CallbackQuery):
    edit_or_send(cb, "👌 Хорошо, история осталась на месте.", None)
    bot.answer_callback_query(cb.id)


# ---------- прочее ----------
@bot.message_handler(content_types=["text"])
def fallback(m: Message):
    bot.send_message(m.chat.id, "📸 Отправь мне <b>фотографию</b> с QR-кодом — я его вырежу!\n"
                                "Если нужна помощь — /help", reply_markup=main_kb())


# ═══════════════════════════════════════════════════════════════════
# 6. ЗАПУСК
# ═══════════════════════════════════════════════════════════════════

import http.server
import socketserver

def _run_health_server():
    port = int(os.getenv("PORT", "7860"))
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"QR Cutter Bot is active!")
        def log_message(self, format, *args):
            pass
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            httpd.serve_forever()
    except Exception:
        pass

threading.Thread(target=_run_health_server, daemon=True).start()

def main():
    me = bot.get_me()
    try:
        bot.set_my_commands([
            BotCommand("start", "🏠 Главное меню"),
            BotCommand("profile", "🍑 Профиль"),
        ])
    except ApiTelegramException:
        log.warning("Не удалось установить список команд")
    bot.remove_webhook()
    log.info("Бот @%s запущен. База: %s", me.username, DB_PATH)
    # infinity_polling сам переподключается после сетевых ошибок
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Бот остановлен")
