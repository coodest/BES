"""Generate assets/bes_demo.gif — 2-class BES boundary shaping animation."""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image



# part 1: generate individual frames as SVGs (with transparent background)

np.random.seed(42)

# ── palette ───────────────────────────────────────────────────────────────────
GRID  = '#21262D'
TEXT  = '#E6EDF3'
SUB   = '#8B949E'
CLR   = ['#FF6B6B', '#4ECDC4']
DCLR  = ['#C0392B', '#1ABC9C']

# ── data: two irregular, overlapping Gaussian clusters ───────────────────────
N = 80

def rotated_cov(sx, sy, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([sx**2, sy**2]) @ R.T

center0 = np.array([-1.8,  0.4])
center1 = np.array([ 1.8, -0.4])
cov0 = rotated_cov(1.5, 0.65,  np.pi / 5)
cov1 = rotated_cov(1.2, 0.80, -np.pi / 6)

pts0 = np.random.multivariate_normal(center0, cov0, N)
pts1 = np.random.multivariate_normal(center1, cov1, N)
pts_init = np.vstack([pts0, pts1])
lbs = np.array([0]*N + [1]*N)

# Final positions: tighter clusters pushed further apart
def tighten_push(pts, ctr, tighten=0.45, push=0.55):
    d = ctr / (np.linalg.norm(ctr) + 1e-9)
    return ctr + (pts - ctr) * tighten + d * push

pts_final = np.vstack([tighten_push(pts0, center0), tighten_push(pts1, center1)])

# ── boundary detection: Gaussian slab criterion ───────────────────────────────
midpoint = (center0 + center1) / 2
dir_raw  = center1 - center0
dir_norm = dir_raw / np.linalg.norm(dir_raw)
proj     = (pts_init - midpoint) @ dir_norm
DELTA    = 1.1  # slab half-width threshold

bnd = np.abs(proj) < DELTA
b_idx = np.where(bnd)[0]
if len(b_idx) > 30:
    b_idx = b_idx[np.random.choice(len(b_idx), 30, replace=False)]
    bnd[:] = False
    bnd[b_idx] = True
nb = ~bnd

# Split boundary nodes into 3 waves
shuf = b_idx.copy()
np.random.shuffle(shuf)
w = len(shuf) // 3
WAVES = [shuf[:w], shuf[w:2*w], shuf[2*w:]]

# ── per-point sinusoidal drift: smooth organic motion during "training" ───────
rng_state = np.random.get_state()
np.random.seed(99)
n_pts   = len(pts_init)
_d_freq = np.random.uniform(0.05, 0.13, (n_pts, 2))
_d_ph   = np.random.uniform(0, 2*np.pi, (n_pts, 2))
_d_amp  = np.random.uniform(0.04, 0.10, (n_pts, 2))
np.random.set_state(rng_state)

def drift(frame, scale):
    if scale <= 0:
        return np.zeros((n_pts, 2))
    return scale * _d_amp * np.sin(frame * _d_freq + _d_ph)

# ── phases ─────────────────────────────────────────────────────────────────────
PHASES = [
    (0,   22, "[1/5]  Initial GNN Embeddings",
              "Spurious correlations entangle nodes near class boundaries"),
    (23,  45, "[2/5]  Boundary Region Detection",
              "Gaussian slab  |proj| ≤ δ  locates the decision boundary zone"),
    (46,  62, "[3/5]  Boundary Nodes Identified",
              "Nodes with cross-class neighbours selected  (S(v) > 0.5)"),
    (63,  86, "[4/5]  Gravity Loss  ·  Wave 1/3",
              "First batch pulled toward class centroids  |  all embeddings shift"),
    (87, 110, "[4/5]  Gravity Loss  ·  Wave 2/3",
              "Second batch pulled  |  model parameters continue updating"),
    (111,134, "[4/5]  Gravity Loss  ·  Wave 3/3",
              "Final batch pulled  |  boundary sharpening converges"),
    (135,154, "[5/5]  Disentangled Embeddings",
              "Structural noise suppressed  |  decision boundaries sharpened"),
]
HOLD_END = 169
FADE_END = 184
N_FRAMES = 185
FPS      = 20
PFADE    = 99  # sentinel for fade-out segment

def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def get_phase(f):
    if f > HOLD_END:
        return PFADE, smooth((f - HOLD_END) / max(FADE_END - HOLD_END, 1))
    for i, (s, e, *_) in enumerate(PHASES):
        if s <= f <= e:
            return i, smooth((f - s) / max(e - s, 1))
    return len(PHASES) - 1, 1.0  # hold period after last phase ends

def get_pts(phase, t, frame):
    if phase <= 2:
        return pts_init.copy()

    wv = phase - 3 if phase <= 5 else 3  # wave index (3 = all waves done)

    if wv < 3:
        cur = pts_init.copy()
        for w in range(wv):                # completed waves: stay at final pos
            cur[WAVES[w]] = pts_final[WAVES[w]]
        cur[WAVES[wv]] = (                 # current wave: interpolate
            pts_init[WAVES[wv]] * (1 - t) + pts_final[WAVES[wv]] * t
        )
        d_scale = smooth(min(1.0, t * 1.5)) if wv == 0 else 1.0
    else:
        # Phase 6 / hold: nb nodes settle toward final pos, drift decays
        cur = pts_final.copy()
        cur[nb] = pts_init[nb] * (1 - t) + pts_final[nb] * t
        d_scale = 1.0 - t

    cur += drift(frame, d_scale)
    return cur

# ── figure: transparent background ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 7.4))
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
ax.set_facecolor('none')
for sp in ax.spines.values():
    sp.set_color('#30363D')
ax.tick_params(colors=SUB, labelsize=7)
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.set_aspect('equal')
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)

leg_h = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CLR[i],
                markersize=9, label=f'Class {i+1}', ls='None') for i in range(2)]
ax.legend(handles=leg_h, loc='upper right',
          framealpha=0, facecolor='none', edgecolor='none',
          labelcolor=TEXT, fontsize=8.5)

ttl     = ax.text(0.5, 1.035, '', transform=ax.transAxes, ha='center',
                  fontsize=13, fontweight='bold', color=TEXT, fontfamily='monospace')
sub_txt = ax.text(0.5, -0.055, '', transform=ax.transAxes, ha='center',
                  fontsize=8.5, color=SUB, fontfamily='monospace')

sc_nb = [ax.scatter([], [], s=22, c=CLR[c], alpha=0, zorder=3, edgecolors='none') for c in range(2)]
sc_b  = [ax.scatter([], [], s=26, c=CLR[c], alpha=0, zorder=4, edgecolors='none') for c in range(2)]
sc_ring = ax.scatter([], [], s=150, facecolors='none', edgecolors='white',
                     lw=1.5, alpha=0, zorder=5)
cpts = [ax.plot([], [], '*', ms=16, color=DCLR[c], zorder=6,
                markeredgecolor='white', markeredgewidth=0.7, alpha=0)[0]
        for c in range(2)]

# Slab ellipse: narrow along dir_raw (width=2δ), wide perpendicular (height=9.5)
slab_angle = np.degrees(np.arctan2(dir_raw[1], dir_raw[0]))
bpatch = Ellipse(midpoint, width=2 * DELTA, height=9.5, angle=slab_angle,
                 color='#4FC3F7', alpha=0, zorder=2)
ax.add_patch(bpatch)

fig.subplots_adjust(bottom=0.17, top=0.96, left=0.09, right=0.97)

# ── progress bar ──────────────────────────────────────────────────────────────
PB_H       = 0.05   # ← 调这个值改变进度条高度 (0–1, relative to pb_ax)
PB_COLOR   = "#949494"
PB_N       = 5
PB_W       = 1.0 / PB_N
PB_PAD     = 0.01
PB_LABELS  = ["Initial Emb.", "Detect Boundary", "Identify", "Refine", "Classification"]
PB_SEG_MAP = [0, 1, 2, 3, 3, 3, 4]
PB_RADIUS  = 0.01   # rounded corner radius

pb_ax = fig.add_axes([0.09, 0.015, 0.88, 0.09])
pb_ax.set_xlim(0, 1)
pb_ax.set_ylim(0, 1)
pb_ax.axis('off')
pb_ax.patch.set_alpha(0)

bar_y = (1 - PB_H) / 2   # vertical centre the bar

def _rr(x, y, w, h, fc, ec, lw=0.0, alpha=1.0, zorder=1):
    r = PB_RADIUS * PB_H
    return FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor=fc, edgecolor=ec, linewidth=lw,
                          alpha=alpha, zorder=zorder)

for i, label in enumerate(PB_LABELS):
    x0 = i * PB_W + PB_PAD
    sw = PB_W - 2 * PB_PAD
    pb_ax.add_patch(_rr(x0, bar_y, sw, PB_H, '#1C2128', '#30363D', lw=0.6))
    # pb_ax.text(x0 + sw / 2, bar_y - 0.18, f"{i+1}/5 · {label}", ha='center', va='top', fontsize=6.5, color='#555C64', fontfamily='monospace')

pb_fills = []
for i in range(PB_N):
    x0 = i * PB_W + PB_PAD
    sw = PB_W - 2 * PB_PAD
    p  = _rr(x0, bar_y, 0.0, PB_H, PB_COLOR, 'none', alpha=0.0, zorder=2)
    pb_ax.add_patch(p)
    pb_fills.append(p)

# ── stateless draw ────────────────────────────────────────────────────────────
def draw_progress(phase, t):
    sw = PB_W - 2 * PB_PAD
    if phase == PFADE:
        fade = 1.0 - t
        for i in range(PB_N):
            pb_fills[i].set_x(i * PB_W + PB_PAD)
            pb_fills[i].set_width(sw)
            pb_fills[i].set_alpha(0.55 * fade)
        return
    seg  = PB_SEG_MAP[phase] if phase < len(PB_SEG_MAP) else PB_N - 1
    frac = (phase - 3 + t) / 3 if 3 <= phase <= 5 else t
    for i in range(PB_N):
        x0 = i * PB_W + PB_PAD
        if i < seg:
            pb_fills[i].set_x(x0); pb_fills[i].set_width(sw); pb_fills[i].set_alpha(0.45)
        elif i == seg:
            pb_fills[i].set_x(x0); pb_fills[i].set_width(sw * frac); pb_fills[i].set_alpha(0.90)
        else:
            pb_fills[i].set_width(0.0); pb_fills[i].set_alpha(0.0)

def draw_frame(frame):
    phase, t = get_phase(frame)

    # Fade-out: everything → alpha 0 for seamless loop restart
    if phase == PFADE:
        fade = 1.0 - t
        ttl.set_alpha(fade)
        sub_txt.set_alpha(fade)
        for c in range(2):
            sc_nb[c].set_alpha(fade * 0.72)
            sc_b[c].set_alpha(fade * 0.80)
            cpts[c].set_alpha(fade * 0.5)
        sc_ring.set_alpha(0)
        bpatch.set_alpha(0)
        draw_progress(phase, t)
        return

    _, _, title, subtitle = PHASES[phase]
    # ttl.set_text(title);       ttl.set_alpha(1.0)
    sub_txt.set_text(subtitle); sub_txt.set_alpha(1.0)

    cur = get_pts(phase, t, frame)
    na = smooth(min(1.0, frame / max(PHASES[0][1], 1) * 1.8)) if phase == 0 else 1.0

    for c in range(2):
        mask = lbs == c
        sc_nb[c].set_offsets(cur[nb & mask])
        sc_nb[c].set_alpha(na * 0.72)
        sc_b[c].set_offsets(cur[bnd & mask])
        sc_b[c].set_alpha(na * 0.80)

    if phase == 0:
        bpatch.set_alpha(0)
        sc_ring.set_alpha(0)
        for cp in cpts:
            cp.set_alpha(0)

    elif phase == 1:
        bpatch.set_alpha(t * 0.12)
        sc_ring.set_alpha(0)
        for c, cp in enumerate(cpts):
            ctr = center0 if c == 0 else center1
            cp.set_data([ctr[0]], [ctr[1]])
            cp.set_alpha(smooth(t * 1.5))

    elif phase == 2:
        bpatch.set_alpha(0.12)
        sc_ring.set_offsets(cur[bnd])
        sc_ring.set_alpha(t * 0.85)
        sc_ring.set_sizes(np.full(bnd.sum(), 140 + t * 55))

    elif 3 <= phase <= 5:
        bpatch.set_alpha(0.08)
        # Only the currently-pulled wave gets rings; rings follow the moving nodes
        ring_mask = np.zeros(n_pts, bool)
        ring_mask[WAVES[phase - 3]] = True
        sc_ring.set_offsets(cur[ring_mask])
        sc_ring.set_sizes(np.full(ring_mask.sum(), 150 + 20 * np.sin(frame * 0.4)))
        sc_ring.set_alpha(0.85)

    elif phase == 6:
        bpatch.set_alpha(max(0.0, 0.08 * (1 - t)))
        sc_ring.set_alpha(max(0.0, 0.85 * (1 - t)))
        for c, cp in enumerate(cpts):
            cp.set_alpha(max(0.0, 1.0 - t * 0.5))

    draw_progress(phase, t)

def init():
    for s in sc_nb + sc_b:
        s.set_offsets(np.empty((0, 2)))
    sc_ring.set_offsets(np.empty((0, 2)))
    for cp in cpts:
        cp.set_data([], [])
    bpatch.set_alpha(0)
    ttl.set_alpha(0)
    sub_txt.set_alpha(0)
    for r in pb_fills:
        r.set_width(0); r.set_alpha(0)
    return []

def update(frame):
    draw_frame(frame)
    return []


os.makedirs('assets/bes_demo', exist_ok=True)
for i in range(N_FRAMES):
    if i == 0:
        init()
    update(i)
    plt.savefig(f"assets/bes_demo/frame_{i:03d}.svg", format="svg", bbox_inches="tight")


plt.close()




































# part 2: combine all frames into a single SVG with CSS animation


import os
import re

# ── 配置 ──────────────────────────────────────────────────────────────────────
FRAME_DIR      = "./assets/bes_demo"
OUTPUT_SVG     = "assets/bes_demo.svg"
FPS            = 20


def extract_inner(svg_text: str) -> str:
    """Strip XML header / DOCTYPE / outer <svg> tag / <metadata>; return inner content."""
    # Skip everything before the opening <svg tag, then skip the tag itself
    svg_idx = svg_text.index('<svg')
    tag_end = svg_text.index('>', svg_idx) + 1
    inner   = svg_text[tag_end:]
    # Drop closing </svg>
    inner   = re.sub(r'\s*</svg>\s*$', '', inner)
    # Drop <metadata>...</metadata>
    inner   = re.sub(r'<metadata>.*?</metadata>', '', inner, flags=re.DOTALL)
    return inner.strip()


def rename_ids(text: str, prefix: str) -> str:
    """Prefix every id attribute and every reference to it in the SVG text."""
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    # Process longest IDs first to avoid partial-string collisions
    for old in sorted(ids, key=len, reverse=True):
        new = f"{prefix}_{old}"
        text = text.replace(f'id="{old}"',            f'id="{new}"')
        text = text.replace(f'url(#{old})',            f'url(#{new})')
        text = text.replace(f'href="#{old}"',          f'href="#{new}"')
        text = text.replace(f'xlink:href="#{old}"',    f'xlink:href="#{new}"')
    return text


# ── 读取所有帧 ─────────────────────────────────────────────────────────────────
frame_files = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith('.svg'))
N  = len(frame_files)
T  = N / FPS          # total cycle duration (s)
dt = 1.0 / FPS        # single-frame duration (s)

# Pick up SVG dimensions from the first frame
with open(os.path.join(FRAME_DIR, frame_files[0]), encoding='utf-8') as fh:
    _first = fh.read()
_svg_tag = _first[_first.index('<svg'): _first.index('>', _first.index('<svg')) + 1]
width    = re.search(r'width="([^"]*)"',   _svg_tag).group(1)
height   = re.search(r'height="([^"]*)"',  _svg_tag).group(1)
viewbox  = re.search(r'viewBox="([^"]*)"', _svg_tag).group(1)

# ── CSS 动画 ────────────────────────────────────────────────────────────────────
# Each frame group has class="frame" (visibility: hidden by default).
# Each frame animates with the same @keyframes _s (total cycle = T seconds),
# but with an increasing animation-delay so they play in sequence.
#
# Keyframes keep the frame visible for exactly the first dt/T fraction of the
# cycle, then hidden for the rest.  Two nearly-adjacent stops bracket the
# transition so the discrete visibility flips exactly at dt/T * T = dt.
pct_lo = 100.0 * dt / T          # ≈ 0.5405 %  (last "visible" stop)
pct_hi = pct_lo + 0.0002         # tiny gap so CSS sees a discrete switch

css_rules = [
    ".frame{visibility:hidden}",
    "@keyframes _s{",
    f"  0%,{pct_lo:.4f}%{{visibility:visible}}",
    f"  {pct_hi:.4f}%,100%{{visibility:hidden}}",
    "}",
]
for i in range(N):
    css_rules.append(
        f"#f{i:03d}{{animation:_s {T:.4f}s linear {i * dt:.4f}s infinite}}"
    )

css_block = (
    "  <defs><style>\n"
    + "\n".join(f"    {r}" for r in css_rules)
    + "\n  </style></defs>"
)

# ── 构建帧 <g> 元素 ──────────────────────────────────────────────────────────────
groups = []
for i, fname in enumerate(frame_files):
    with open(os.path.join(FRAME_DIR, fname), encoding='utf-8') as fh:
        text = fh.read()
    inner = extract_inner(text)
    inner = rename_ids(inner, f"f{i:03d}")
    groups.append(f'  <g id="f{i:03d}" class="frame">\n{inner}\n  </g>')
    if (i + 1) % 20 == 0:
        print(f"  processed {i + 1}/{N} frames …")

# ── 输出 ────────────────────────────────────────────────────────────────────────
svg_out = "\n".join([
    f'<svg xmlns="http://www.w3.org/2000/svg"'
    f' xmlns:xlink="http://www.w3.org/1999/xlink"'
    f' width="{width}" height="{height}" viewBox="{viewbox}">',
    css_block,
    *groups,
    '</svg>',
])

os.makedirs(os.path.dirname(OUTPUT_SVG) or '.', exist_ok=True)
with open(OUTPUT_SVG, 'w', encoding='utf-8') as fh:
    fh.write(svg_out)

size_mb = os.path.getsize(OUTPUT_SVG) / 1024 / 1024
print(f"✓  {N} frames  →  {OUTPUT_SVG}  ({size_mb:.1f} MB)")


shutil.rmtree("assets/bes_demo")