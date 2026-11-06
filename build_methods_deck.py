"""Methods-first revision of the project deck.

Reuses the visual system in build_project_deck.py, but rebalances the narrative:
13 of 20 slides explain the model and protocol; 4 show the SHD evidence.
Run with the bundled Python (python-pptx 1.0.2).
"""

from pathlib import Path
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

from build_project_deck import (
    PAPER, WHITE, INK, INK_SOFT, MUTED, TEAL, TEAL_DARK, TEAL_PALE,
    OCHRE, OCHRE_PALE, RED, RED_PALE, BLUE_GRAY, BORDER, GRID_LIGHT,
    DARK_BG, TITLE_FONT, BODY_FONT, MATH_FONT, SLIDE_W, SLIDE_H,
    MARGIN_L, MARGIN_R, CONTENT_W,
    add_text, add_math, add_line, add_card, add_rect, add_chip,
    add_footer, add_header, set_notes, new_slide, stat_card,
    vector_grid,
)

FIG = Path("figures/manuscript/panels")
OUT = Path("neuron_as_vector_project_deck_v4.pptx")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def add_figure(s, x, y, w, h, name, caption=None, cap_h=0.34):
    """Place a figure inside a box, preserving aspect ratio, with optional caption."""
    path = FIG / name
    iw, ih = Image.open(path).size
    box_h = h - (cap_h if caption else 0)
    scale = min(w / iw, box_h / ih)
    pw, ph = iw * scale, ih * scale
    px, py = x + (w - pw) / 2, y + (box_h - ph) / 2
    add_card(s, x - 0.05, y - 0.05, w + 0.10, box_h + 0.10, fill=WHITE, line=BORDER, radius=0.025)
    s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if caption:
        add_text(s, x, y + box_h + 0.12, w, cap_h, caption, size=8, color=MUTED, line_spacing=1.05)


def raster(s, x, y, w, h, rows=7, color=TEAL):
    """Schematic event raster (illustration, not data)."""
    add_line(s, x, y + h - 0.05, x + w, y + h - 0.05, color=BORDER, width=0.6)
    for r in range(rows):
        yy = y + 0.10 + r * (h - 0.18) / rows
        for k in range(7):
            frac = (k * 0.61803398875 + r * 0.271 + 0.13) % 1.0
            xx = x + 0.04 + frac * (w - 0.08)
            add_rect(s, xx, yy, 0.045, 0.045, color)


def flow_box(s, x, y, w, h, title, sub=None, accent=TEAL):
    add_card(s, x, y, w, h, fill=WHITE)
    add_rect(s, x, y, 0.06, h, accent)
    add_text(s, x + 0.18, y + 0.14, w - 0.28, 0.28, title, size=10.5, color=INK, bold=True)
    if sub:
        add_text(s, x + 0.18, y + 0.43, w - 0.28, h - 0.50, sub, size=8.5, color=MUTED, line_spacing=1.12)


def small_note(s, x, y, w, text, size=8.2, color=MUTED, h=0.25):
    add_text(s, x, y, w, h, text, size=size, color=color, line_spacing=1.1)


def compact_stat(s, x, y, w, h, value, label, sub, accent=TEAL, value_size=21):
    add_card(s, x, y, w, h, fill=WHITE)
    add_rect(s, x, y, 0.06, h, accent)
    add_text(s, x + 0.19, y + 0.11, w - 0.28, 0.2, label.upper(), size=8.0, color=accent, bold=True)
    add_text(s, x + 0.19, y + 0.33, w - 0.28, 0.55, value, size=value_size, color=INK, bold=True)
    add_text(s, x + 0.19, y + 0.95, w - 0.28, h - 1.0, sub, size=8.3, color=MUTED, line_spacing=1.05)


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_title(prs):
    s = new_slide(prs, DARK_BG)
    vector_grid(s, 9.55, 1.35, 3.1, 3.1, rows=8, cols=8, gap=0.055,
                color=RGBColor(0x1C, 0x43, 0x4C), pale=RGBColor(0x16, 0x3A, 0x42))
    add_text(s, 9.55, 4.58, 3.2, 0.25, "one neuron  ·  1,000 coordinates", size=8.5,
             color=RGBColor(0x9F, 0xB3, 0xB4))
    add_text(s, 0.8, 0.68, 5.2, 0.25, "PROJECT OVERVIEW  ·  2026", size=9.5,
             color=TEAL_PALE, bold=True)
    add_text(s, 0.8, 1.35, 8.4, 1.95, "What a neuron's\ninternal state buys",
             size=44, color=WHITE, bold=True, font=TITLE_FONT, line_spacing=0.95)
    add_text(s, 0.82, 3.52, 8.2, 0.85,
             "A methods-first walkthrough of a vector-valued\nspiking neuron on Spiking Heidelberg Digits",
             size=18, color=RGBColor(0xC9, 0xD6, 0xD6), line_spacing=1.15)
    add_text(s, 0.82, 4.72, 7.5, 0.3, "Mehran Jaberinezhad  ·  November 2026", size=12,
             color=RGBColor(0xE8, 0xEF, 0xEF))
    chips = [("N = 100", 0.82, 5.42, 1.15), ("D = 1000", 2.12, 5.42, 1.3),
             ("SHD  ·  20 classes", 3.57, 5.42, 1.95), ("13 methods / 4 results", 5.67, 5.42, 2.35)]
    for text, x, y, w in chips:
        add_chip(s, x, y, w, 0.34, text, fill=RGBColor(0x16, 0x3A, 0x42),
                 color=RGBColor(0xD9, 0xE7, 0xE6), size=9)
    add_text(s, 0.82, 6.05, 8.4, 0.25,
             "REPO  ·  [ paste repository URL here ]", size=9.3,
             color=RGBColor(0xA9, 0xD3, 0xD2))
    add_line(s, 0.82, 6.32, 5.2, 6.32, color=RGBColor(0x2E, 0x5A, 0x62), width=0.6)
    add_text(s, 0.82, 6.55, 8.4, 0.25,
             "Most of the deck is the model; the last four slides are the evidence.",
             size=8.5, color=RGBColor(0x8F, 0xA7, 0xA9))
    add_footer(s, 1, dark=True)
    set_notes(s,
              "The deck is now methods-first: 13 slides on the model and protocol, 4 on the SHD evidence. "
              "The question is whether a genuinely vector-valued state buys computation at fixed neuron count, "
              "readout and budget. The answer is tested on SHD, 20 classes, with speakers 4 and 5 never seen in training.")


def slide_question(prs):
    s = new_slide(prs)
    add_header(s, "THE QUESTION", "A spike is one bit. What if the state is not?")
    add_card(s, 0.7, 1.86, 5.35, 4.55, fill=WHITE)
    add_text(s, 0.95, 2.04, 4.8, 0.25, "STANDARD LIF", size=9, color=OCHRE, bold=True)
    add_math(s, 0.95, 2.38, 4.85, 0.45, "V_i(t+1) = (1 − α) V_i(t) + I_i(t)", size=14.5)
    add_math(s, 0.95, 2.88, 4.85, 0.45, "s_i(t) = H(V_i(t) − θ_i)", size=14.5)
    add_text(s, 0.95, 3.45, 4.8, 1.75,
             "One scalar state. One threshold. One output bit.\n\n"
             "The neuron can integrate, leak and fire. It cannot hold a pattern whose "
             "coordinates vary independently: a spike says only that the scalar crossed threshold.",
             size=12.5, color=INK_SOFT, line_spacing=1.18)
    add_text(s, 0.95, 5.45, 4.8, 0.55, "one number  →  one bit", size=10, color=OCHRE, bold=True)

    add_card(s, 6.3, 1.86, 6.3, 4.55, fill=WHITE)
    add_text(s, 6.55, 2.04, 5.7, 0.25, "THIS MODEL", size=9, color=TEAL, bold=True)
    vector_grid(s, 6.58, 2.42, 1.55, 1.55, rows=6, cols=6, gap=0.05, color=TEAL, pale=TEAL_PALE)
    add_math(s, 8.35, 2.52, 4.0, 0.45, "z_i(t) ∈ ℝ^1000", size=19)
    add_text(s, 8.35, 3.02, 3.95, 1.0,
             "Every coordinate participates in the dynamics. The spike is still binary, but it "
             "emits the whole state vector.", size=11.5, color=INK_SOFT, line_spacing=1.15)
    add_math(s, 6.55, 4.18, 5.7, 0.4, "s_i(t) = H(⟨z_i, w_out,i⟩ + θ_i − w_ad b_i)", size=13.5)
    add_text(s, 6.55, 4.72, 5.6, 1.4,
             "The readout is a learned hyperplane through the state, not a scalar membrane. "
             "The width of what a spike can transmit equals the width of the state.",
             size=11.5, color=INK_SOFT, line_spacing=1.15)
    add_card(s, 0.7, 6.52, 11.9, 0.0, fill=PAPER, line=PAPER)
    add_chip(s, 0.7, 6.44, 3.9, 0.32, "fixed N  ·  fixed readout  ·  fixed budget", fill=TEAL_PALE, color=TEAL_DARK, size=8.5)
    add_text(s, 4.78, 6.45, 7.8, 0.32, "Does state width itself buy computation?", size=12.5, color=TEAL_DARK, bold=True)
    add_footer(s, 2, source="The comparison is architectural; the empirical test is on SHD (next slides).")
    set_notes(s,
              "The scalar neuron is not a straw man: it is the standard event-driven unit. "
              "The vector model keeps the same spike interface and the same count readout; only the state and the "
              "intra-neuron operations become vector-valued. The controlled question is stated at the bottom.")


def slide_shd(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · BENCHMARK", "SHD: 20 words, 700 channels, one second")
    cards = [
        ("8,156", "training trials", "speakers 0–3, 6–11", TEAL),
        ("2,264", "official test trials", "speakers 4 & 5 unseen", OCHRE),
        ("20", "German digits", "classes 0–19", TEAL),
        ("700", "cochlear channels", "binary events", BLUE_GRAY),
    ]
    for i, (v, lab, sub, acc) in enumerate(cards):
        x = 0.7 + (i % 2) * 2.55
        y = 1.9 + (i // 2) * 1.75
        stat_card(s, x, y, 2.4, 1.55, v, lab, sub, accent=acc, value_size=25)

    add_figure(s, 5.85, 1.86, 6.75, 4.55, "fig_activity_test_sample1.png",
               "Reported final run, held-out trial: input events (top), 100-neuron spike raster (middle), "
               "population rate (bottom).", cap_h=0.42)

    add_card(s, 0.7, 5.5, 4.95, 1.15, fill=TEAL_PALE, line=TEAL_PALE)
    add_text(s, 0.92, 5.64, 4.55, 0.25, "WHY SHD", size=8.5, color=TEAL_DARK, bold=True)
    add_text(s, 0.92, 5.9, 4.5, 0.65,
             "Event-based, 20-way, standard split. Accuracy is comparable across models and "
             "the test speakers are truly held out.", size=9.2, color=INK_SOFT, line_spacing=1.1)
    add_footer(s, 3, source="SHD: 8,156 train / 2,264 test; 700 channels. Activity: v3_n100_spec_ep30, held-out trial.")
    set_notes(s,
              "SHD is the standard event-based digit benchmark: 20 German digits, 700 cochlear channels, sparse events. "
              "We bin at 2 ms, which gives 500 bins for the 1 s trial, then append 50 zero bins (100 ms) so the "
              "count readout sees the tail of the response. The official test set has 2,264 trials; speakers 4 and 5 "
              "never appear in training, which makes it a genuine unseen-speaker test.")


def slide_architecture(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · ARCHITECTURE", "One shared input, 100 vector neurons, one count readout")
    add_figure(s, 0.7, 1.86, 8.45, 4.75, "fig01_schematic.png",
               "Figure 1. Population-vector architecture. A spike emits the whole state; emitted vectors are summed "
               "and fed back through the emission matrix; a linear readout on spike counts produces the class.")
    rail = [
        ("INPUT", "700 channels  →  1,400", "two delay-line copies; shared input matrix"),
        ("NEURON", "100 × 1,000 state", "rank-64 mixing; learned leak"),
        ("READOUT", "100 spikes  →  20", "counts over 550 bins; classifier"),
    ]
    for i, (lab, big, sub) in enumerate(rail):
        y = 1.86 + i * 1.55
        add_card(s, 9.35, y, 3.25, 1.35, fill=WHITE)
        add_text(s, 9.55, y + 0.12, 2.9, 0.22, lab, size=8.2, color=TEAL, bold=True)
        add_text(s, 9.55, y + 0.36, 2.9, 0.32, big, size=13.5, color=INK, bold=True)
        add_text(s, 9.55, y + 0.72, 2.9, 0.5, sub, size=8.2, color=MUTED, line_spacing=1.1)
    add_text(s, 9.35, 6.5, 3.25, 0.3, "3,069,201 parameters", size=9, color=OCHRE, bold=True)
    add_footer(s, 4, source="All blocks drawn to scale in width, not in parameter count; see Methods §2.4–2.5.")
    set_notes(s,
              "The figure is the whole architecture. The input is 700 channels, expanded to 1,400 by the fixed "
              "delay line, then projected by one shared 1,400×1,000 matrix. Each of the 100 neurons carries a "
              "1,000-dimensional state and mixes it with a rank-64 operator. A spike emits the whole state; the "
              "emitted vectors are summed and passed through a shared 1,000×1,000 W_emit. The readout is a linear "
              "classifier on spike counts over the 550 bins.")


def slide_input(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · WHERE THE VECTOR ENTERS · 1/3", "The input projection makes the vector")
    add_card(s, 0.7, 1.86, 6.55, 3.4, fill=WHITE)
    add_text(s, 0.95, 2.04, 5.9, 0.25, "SHARED PROJECTION + PER-NEURON GATE", size=8.8, color=TEAL, bold=True)
    add_math(s, 0.95, 2.42, 6.0, 0.5, "a_t = x_t W_in", size=22)
    add_math(s, 0.95, 2.98, 6.0, 0.5, "d_in,i,t = g_i ⊙ a_t", size=20)
    add_text(s, 0.95, 3.62, 5.95, 1.35,
             "The input is a 1×1,400 row; the input matrix is 1,400×1,000. The result is one 1,000-D pattern "
             "shared by all neurons. The gate is a learned 1,000-D mask per neuron.",
             size=11.5, color=INK_SOFT, line_spacing=1.15)
    add_chip(s, 0.95, 4.75, 1.95, 0.3, "input matrix: 1.4M", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 3.02, 4.75, 1.45, 0.3, "gate: 100k", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    # Matrix schematic
    add_card(s, 7.55, 1.86, 5.05, 3.4, fill=WHITE)
    add_text(s, 7.78, 2.04, 4.6, 0.25, "SHAPES", size=8.8, color=MUTED, bold=True)
    boxes = [("x(t)", "1 × 1400", 7.8, 2.45, 1.15, 0.7, OCHRE_PALE, OCHRE),
             ("Wᵢₙ", "1400 × 1000", 9.15, 2.45, 1.35, 0.7, TEAL_PALE, TEAL),
             ("a(t)", "1 × 1000", 10.7, 2.45, 1.2, 0.7, TEAL_PALE, TEAL)]
    for lab, sub, x, y, w, h, fill, line in boxes:
        add_card(s, x, y, w, h, fill=fill, line=line, radius=0.04)
        add_text(s, x + 0.08, y + 0.10, w - 0.16, 0.3, lab, size=13, color=INK, bold=True, align=PP_ALIGN.CENTER)
        add_text(s, x + 0.06, y + 0.42, w - 0.12, 0.2, sub, size=7.8, color=MUTED, align=PP_ALIGN.CENTER)
    add_line(s, 8.98, 2.8, 9.1, 2.8, color=INK_SOFT, width=1.0, arrow=True)
    add_line(s, 10.53, 2.8, 10.65, 2.8, color=INK_SOFT, width=1.0, arrow=True)
    add_text(s, 7.8, 3.35, 4.6, 0.45, "one projection, reused by all 100 neurons", size=8.5, color=MUTED)
    add_math(s, 7.8, 3.92, 4.5, 0.4, "g_i ⊙ a_t", size=16, color=TEAL_DARK)
    add_text(s, 7.8, 4.38, 4.55, 0.7, "the gate selects coordinates; it does not mix them", size=8.8, color=MUTED)

    add_card(s, 0.7, 5.48, 11.9, 1.2, fill=TEAL_PALE, line=TEAL_PALE)
    add_text(s, 0.95, 5.63, 11.35, 0.25, "WHY THIS IS THE FIRST INSERTION POINT", size=8.5, color=TEAL_DARK, bold=True)
    add_text(s, 0.95, 5.9, 11.35, 0.65,
             "A scalar neuron would have one weight vector and produce one current. Here a shared matrix produces a "
             "1,000-D pattern; the gate makes that pattern neuron-specific. Parameter cost is 1,400×1,000, not 100 times that.",
             size=10.2, color=INK_SOFT, line_spacing=1.12)
    add_footer(s, 5, source="Methods §2.4.1–2.4.2; two delay-line copies give 1,400 input channels.")
    set_notes(s,
              "The first place the vector enters is the input projection. Every neuron does not own a 1,400×1,000 "
              "matrix; there is one shared W_in, and each neuron has a learned 1,000-D gate. The gate is coordinate-wise, "
              "so the input path creates a vector but does not mix its coordinates.")


def slide_mixing(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · WHERE THE VECTOR ENTERS · 2/3", "Rank-64 mixing inside each neuron")
    add_card(s, 0.7, 1.86, 6.5, 3.75, fill=WHITE)
    add_text(s, 0.95, 2.04, 5.8, 0.25, "ONE LOW-RANK OPERATOR, SHARED BY ALL NEURONS", size=8.8, color=TEAL, bold=True)
    add_math(s, 0.95, 2.45, 6.0, 0.55, "m_i = (z_i W_down) W_up", size=22)
    add_text(s, 0.95, 3.12, 5.95, 1.65,
             "The two factors are 1,000×64 and 64×1,000; both are shared. The 64-D bottleneck is the only place where "
             "coordinates of one neuron's state interact. A dense 1,000×1,000 map per neuron would cost 100M "
             "parameters; this costs 128k.",
             size=11.5, color=INK_SOFT, line_spacing=1.16)
    add_chip(s, 0.95, 4.78, 2.0, 0.3, "rank R = 64", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 3.08, 4.78, 2.25, 0.3, "128k parameters", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    add_card(s, 7.5, 1.86, 5.1, 3.75, fill=WHITE)
    add_text(s, 7.75, 2.04, 4.6, 0.25, "LOW-RANK BOTTLENECK", size=8.8, color=MUTED, bold=True)
    # 1000 -> 64 -> 1000 schematic
    for i, (lab, sub, x) in enumerate([("1000", "state", 7.9), ("64", "rank", 9.65), ("1000", "update", 11.4)]):
        w = 1.2 if i != 1 else 0.85
        add_rect(s, x, 2.55, w, 1.15, TEAL_PALE if i == 1 else WHITE, TEAL if i == 1 else BORDER)
        add_text(s, x, 2.72, w, 0.3, lab, size=14, color=INK, bold=True, align=PP_ALIGN.CENTER)
        add_text(s, x, 3.06, w, 0.2, sub, size=8, color=MUTED, align=PP_ALIGN.CENTER)
    add_line(s, 9.12, 3.1, 9.55, 3.1, color=INK_SOFT, width=1.0, arrow=True)
    add_line(s, 10.52, 3.1, 10.95, 3.1, color=INK_SOFT, width=1.0, arrow=True)
    add_text(s, 7.8, 4.0, 4.55, 0.9,
             "The bottleneck forces a shared low-dimensional subspace. It is an architectural prior, "
             "not a claim that the state is only 64-dimensional.",
             size=9.2, color=INK_SOFT, line_spacing=1.12)

    add_card(s, 0.7, 5.85, 11.9, 0.85, fill=OCHRE_PALE, line=OCHRE_PALE)
    add_text(s, 0.95, 5.98, 11.35, 0.55,
             "This is the only active mechanism that mixes coordinates within a single neuron. The emission matrix "
             "mixes across neurons; the gate and the readout are coordinate-wise or linear.",
             size=10.2, color=INK_SOFT, line_spacing=1.12)
    add_footer(s, 6, source="Methods §2.4.4; both factors initialised from zero-mean Gaussians (1/D and 0.25/R).")
    set_notes(s,
              "The second insertion point is the intrinsic mixing. Each neuron multiplies its own 1,000-D state by a "
              "shared 1,000×64 matrix, then by a shared 64×1,000 matrix. The rank is 64, so the operator is deliberately "
              "low-rank. This is the only active mechanism that mixes coordinates inside one neuron.")


def slide_broadcast(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · WHERE THE VECTOR ENTERS · 3/3", "A spike broadcasts the whole vector")
    add_card(s, 0.7, 1.86, 6.55, 3.55, fill=WHITE)
    add_text(s, 0.95, 2.04, 5.9, 0.25, "RECURRENT COUPLING + SHARED EMISSION", size=8.8, color=TEAL, bold=True)
    add_math(s, 0.95, 2.42, 6.0, 0.45, "c_i(t) = Σ_j W_rec[j,i] s_j(t)", size=17)
    add_math(s, 0.95, 2.92, 6.0, 0.45, "h(t) = (1/√N)(Σ_j s_j(t) z_j(t)) W_emit", size=15.5)
    add_math(s, 0.95, 3.45, 6.0, 0.45, "d_rec,i = c_i(t) h(t)", size=19)
    add_text(s, 0.95, 4.02, 5.95, 1.3,
             "A spike is one scalar; the state is 1,000-D. A single spike therefore injects the whole state of neuron j "
             "into a shared 1,000-D population signal, scaled by the scalar coupling.",
             size=11.2, color=INK_SOFT, line_spacing=1.14)
    add_chip(s, 0.95, 4.88, 1.75, 0.3, "recurrent: 10k", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 2.8, 4.88, 1.8, 0.3, "emission: 1M", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    add_card(s, 7.55, 1.86, 5.05, 3.55, fill=WHITE)
    add_text(s, 7.78, 2.04, 4.55, 0.25, "ONE BIT  →  ONE VECTOR", size=8.8, color=MUTED, bold=True)
    # three neurons with vectors, summed
    for i, yy in enumerate([2.35, 3.0, 3.65]):
        vector_grid(s, 7.85, yy, 0.55, 0.55, rows=3, cols=3, gap=0.025, color=TEAL, pale=TEAL_PALE)
        add_text(s, 8.52, yy + 0.08, 0.9, 0.25, f"s{chr(0x2080 + i + 1)} = 1", size=8.5, color=INK_SOFT)
        add_line(s, 9.35, yy + 0.27, 9.95, 3.35, color=BORDER, width=0.8)
    add_rect(s, 9.95, 2.78, 0.9, 1.1, TEAL_PALE, TEAL)
    add_text(s, 9.98, 3.08, 0.84, 0.3, "Σ", size=18, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_line(s, 10.9, 3.35, 11.35, 3.35, color=INK_SOFT, width=1.0, arrow=True)
    add_rect(s, 11.35, 2.9, 0.85, 0.9, WHITE, OCHRE)
    add_text(s, 11.35, 3.06, 0.85, 0.36, "W", size=18, color=OCHRE, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, 11.35, 3.44, 0.85, 0.2, "emit", size=7.2, color=OCHRE, align=PP_ALIGN.CENTER)
    add_text(s, 7.8, 4.3, 4.55, 0.85,
             "The sum is a 1,000-D population vector; one shared matrix maps it back to state space and is used by every neuron.",
             size=8.8, color=INK_SOFT, line_spacing=1.12)

    add_card(s, 0.7, 5.65, 11.9, 1.05, fill=TEAL_DARK, line=TEAL_DARK)
    add_text(s, 0.95, 5.78, 11.35, 0.25, "WHY THE 1/√N", size=8.5, color=RGBColor(0xA9, 0xD3, 0xD2), bold=True)
    add_text(s, 0.95, 6.03, 11.35, 0.55,
             "The sum over N neurons grows with N; the 1/√N keeps the emitted vector's scale roughly constant, so the "
             "learned dynamics do not have to re-tune when the population size changes.",
             size=9.8, color=WHITE, line_spacing=1.1)
    add_footer(s, 7, source="Methods §2.4.3; the emission matrix is initialised as identity + 0.02 noise, 1,000×1,000.")
    set_notes(s,
              "The third insertion point is the broadcast. When neuron j spikes, the scalar coupling c_i multiplies a "
              "shared 1,000-D vector h, which is built by summing the states of all neurons that spiked, weighted by "
              "their spike, and passing through W_emit. That is how one bit can carry a whole state vector. The 1/sqrt(N) "
              "keeps the scale from growing with N.")


def slide_state(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · DYNAMICS", "The state update and the spike functional")
    add_card(s, 0.7, 1.86, 6.7, 3.95, fill=WHITE)
    add_text(s, 0.95, 2.04, 6.0, 0.25, "LEAK, TWO-STAGE TARGET, SPIKE", size=8.8, color=TEAL, bold=True)
    add_math(s, 0.95, 2.42, 6.2, 0.45, "z_i(t+1) = (1 − α_i) z_i(t) + α_i z̃_i(t)", size=17)
    add_math(s, 0.95, 2.95, 6.2, 0.45, "z̃_i = tanh(m_i + λ tanh(m_i^(2) + d_i))", size=16)
    add_math(s, 0.95, 3.48, 6.2, 0.45, "o_i = ⟨z_i, w_out,i⟩ + θ_i − w_ad b_i", size=17)
    add_math(s, 0.95, 4.02, 6.2, 0.45, "s_i = H(o_i)", size=18)
    add_text(s, 0.95, 4.62, 6.1, 1.0,
             "The leak is learned per neuron; λ is a learned scalar mixing the two rank-64 paths; θ is the "
             "calibrated threshold; the readout vector is a learned hyperplane.",
             size=10.2, color=INK_SOFT, line_spacing=1.12)

    add_card(s, 7.65, 1.86, 4.95, 3.95, fill=WHITE)
    add_text(s, 7.88, 2.04, 4.4, 0.25, "THE VECTOR IS READ BY A HYPERPLANE", size=8.8, color=MUTED, bold=True)
    vector_grid(s, 7.9, 2.42, 1.2, 1.2, rows=5, cols=5, gap=0.035, color=TEAL, pale=TEAL_PALE)
    add_math(s, 9.3, 2.55, 3.1, 0.4, "z_i ∈ ℝ^1000", size=14)
    add_math(s, 9.3, 3.0, 3.1, 0.4, "w_out,i ∈ ℝ^1000", size=13)
    add_text(s, 7.9, 3.95, 4.45, 1.4,
             "The spike is still one bit. What changed is that the decision boundary is a learned hyperplane through "
             "1,000 coordinates, not a threshold on one membrane potential.",
             size=9.5, color=INK_SOFT, line_spacing=1.14)

    add_card(s, 0.7, 6.0, 11.9, 0.72, fill=TEAL_PALE, line=TEAL_PALE)
    add_text(s, 0.95, 6.12, 11.35, 0.45,
             "Reported configuration: leak init 0.1, λ init 1.0, thresholds calibrated to ~10 Hz, readout weights ~ N(0, 1/D). "
             "adLIF currents are configured but do not enter the reported two-stage target.",
             size=9.4, color=INK_SOFT, line_spacing=1.1)
    add_footer(s, 8, source="Methods §2.4.5–2.4.8; two-stage target Eq. 30; spike functional Eq. 23.")
    set_notes(s,
              "The state update is a leaky integrator with a learned per-neuron leak. The target is the two-stage "
              "nonlinearity: a rank-64 mixing, a second rank-64 path, and a tanh. The spike is a Heaviside of a learned "
              "hyperplane through the whole state. That is the only place the full vector is read.")


def slide_forward(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · FORWARD PASS", "One trial, end to end")
    add_card(s, 0.7, 1.86, 7.3, 4.9, fill=WHITE)
    add_text(s, 0.95, 2.02, 6.8, 0.25, "ONE BIN, IN ORDER", size=8.8, color=TEAL, bold=True)
    lines = [
        "a_t   = x_t W_in",
        "d_in  = g ⊙ a_t",
        "m     = (z W_down) W_up",
        "m2    = (z W_down2) W_up2",
        "c     = s W_rec",
        "h     = (1/√N)(Σ_j s_j z_j) W_emit",
        "d     = d_in + c h + β",
        "z̃     = tanh(m + λ tanh(m2 + d))",
        "z     ← (1 − α) z + α z̃",
        "o     = ⟨z, w_out⟩ + θ − w_ad b",
        "s     = H(o)",
    ]
    for i, line in enumerate(lines):
        add_text(s, 0.98, 2.36 + i * 0.37, 6.8, 0.3, line, size=12.5, color=INK, font="Consolas")

    add_card(s, 8.25, 1.86, 4.35, 4.9, fill=WHITE)
    add_text(s, 8.48, 2.02, 3.9, 0.25, "AROUND THE LOOP", size=8.8, color=MUTED, bold=True)
    bullets = [
        ("T = 550", "500 signal bins + 50 silence bins"),
        ("10 chunks", "checkpoint every 55 bins"),
        ("fp16 state", "fp32 master weights; GradScaler"),
        ("counts", "sum spikes over t; W_cls → 20"),
    ]
    for i, (big, sub) in enumerate(bullets):
        y = 2.45 + i * 1.02
        add_text(s, 8.48, y, 1.35, 0.3, big, size=13, color=TEAL_DARK, bold=True)
        add_text(s, 8.48, y + 0.31, 3.8, 0.5, sub, size=8.6, color=MUTED, line_spacing=1.1)

    add_footer(s, 9, source="Algorithm 1 in Methods §2.5; grad checkpointing in §2.8.4.")
    set_notes(s,
              "This is the per-bin forward pass. The delay line and input projection happen once for the whole trial; "
              "the loop then updates the state, computes the spike, and accumulates counts. The right rail gives the "
              "loop-level facts: 550 bins, ten checkpointed chunks, fp16 state with fp32 masters, and a count readout.")


def slide_init(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · INITIALISATION + CALIBRATION", "Start near 10 Hz, not at silence")
    # parameter budget cards
    stats = [
        ("3,069,201", "total parameters", "19 groups", TEAL),
        ("2.80M", "input + emission", "91% of the budget", OCHRE),
        ("256k", "rank-64 mixers", "two stages", BLUE_GRAY),
        ("12k", "recurrent + readout", "classifier and thresholds", TEAL),
    ]
    for i, (v, lab, sub, acc) in enumerate(stats):
        x = 0.7 + (i % 2) * 2.9
        y = 1.9 + (i // 2) * 1.6
        compact_stat(s, x, y, 2.72, 1.5, v, lab, sub, accent=acc, value_size=20)

    add_card(s, 6.6, 1.86, 6.0, 4.55, fill=WHITE)
    add_text(s, 6.85, 2.02, 5.5, 0.25, "THRESHOLD CALIBRATION", size=8.8, color=TEAL, bold=True)
    add_math(s, 6.85, 2.40, 5.4, 0.4, "p = clip(10 Hz × 2 ms, 0.001, 0.49) = 0.02", size=12.5)
    add_text(s, 6.85, 2.85, 5.45, 0.7,
             "Set all thresholds to zero. Run three fixed-point iterations on 96 FIT trials:", size=9.6, color=INK_SOFT)
    add_math(s, 6.98, 3.35, 5.3, 0.4, "θ_i  ←  θ_i − Q_(1−p)(o_i)", size=17)
    add_text(s, 6.85, 3.86, 5.45, 0.9,
             "Skip the first 68 bins, flatten over time and batch, and take the (1−p) quantile of the "
             "pre-activation values. Three iterations land near the 2%-per-bin target.",
             size=9.2, color=INK_SOFT, line_spacing=1.12)
    add_chip(s, 6.85, 4.88, 2.0, 0.3, "2% per bin ≈ 10 Hz", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_text(s, 6.85, 5.35, 5.4, 0.8,
             "The thresholds stay trainable after calibration. Calibration sets the operating point; training can move it.",
             size=9.0, color=MUTED, line_spacing=1.12)

    add_card(s, 0.7, 5.3, 5.65, 1.2, fill=OCHRE_PALE, line=OCHRE_PALE)
    add_text(s, 0.95, 5.44, 5.15, 0.25, "WHY CALIBRATE", size=8.5, color=OCHRE, bold=True)
    add_text(s, 0.95, 5.69, 5.15, 0.7,
             "A count readout needs spikes. Zero thresholds either silence the population or saturate it; "
             "10 Hz is a stable operating point across neurons.", size=9.0, color=INK_SOFT, line_spacing=1.1)
    add_footer(s, 10, source="Methods §2.6; parameter groups Table 4; calibration Algorithm 2.")
    set_notes(s,
              "Initialisation matters because the readout is a count. We calibrate every threshold to a target of about "
              "10 Hz: p is the target spike probability per 2 ms bin, 0.02. Three fixed-point iterations on the "
              "pre-activation quantile put each neuron near that rate. The threshold remains trainable.")


def slide_mechanisms1(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · MECHANISMS · 1/2", "Memory beyond the membrane")
    # adaptation
    add_card(s, 0.7, 1.86, 5.75, 3.75, fill=WHITE)
    add_text(s, 0.95, 2.04, 5.2, 0.25, "SPIKE-FREQUENCY ADAPTATION", size=8.8, color=TEAL, bold=True)
    add_math(s, 0.95, 2.42, 5.3, 0.4, "b_i(t+1) = ρ b_i(t) + s_i(t)", size=16)
    add_math(s, 0.95, 2.9, 5.3, 0.4, "o_i  ←  o_i − w_ad b_i", size=16)
    add_text(s, 0.95, 3.42, 5.3, 1.0,
             "A soft spike-history term. ρ = 0.95; the adaptation weight starts at 0.1 and is learned. It replaces "
             "reset/refractory: the neuron is slowed after firing, not blanked.",
             size=10.0, color=INK_SOFT, line_spacing=1.14)
    add_chip(s, 0.95, 4.62, 1.55, 0.3, "ρ = 0.95", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 2.62, 4.62, 1.95, 0.3, "adaptation: 0.1", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    # state dynamics
    add_card(s, 6.55, 1.86, 6.05, 3.75, fill=WHITE)
    add_text(s, 6.8, 2.04, 5.5, 0.25, "LEARNABLE LEAK + INITIAL STATE", size=8.8, color=TEAL, bold=True)
    add_math(s, 6.8, 2.42, 5.55, 0.4, "α_i = sigmoid(raw_i),  init 0.1", size=15)
    add_math(s, 6.8, 2.9, 5.55, 0.4, "z_i(0) = z0_i  (learned, 100k)", size=13.5)
    add_text(s, 6.8, 3.42, 5.55, 1.0,
             "Each neuron gets its own leak; the population is not forced to share one time constant. "
             "The initial state is a learned 1,000-D vector per neuron, not zeros.",
             size=10.0, color=INK_SOFT, line_spacing=1.14)
    add_chip(s, 6.8, 4.62, 1.9, 0.3, "100 leak values", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 8.82, 4.62, 2.05, 0.3, "100k initial state", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    add_card(s, 0.7, 5.85, 11.9, 0.85, fill=TEAL_PALE, line=TEAL_PALE)
    add_text(s, 0.95, 5.98, 11.35, 0.55,
             "Both are small: adaptation adds 100 weights; the learned initial state adds 100,000. "
             "The 20-epoch screen kept both; they are part of the reported model.",
             size=9.5, color=INK_SOFT, line_spacing=1.1)
    add_footer(s, 11, source="Methods §2.7.1–2.7.2; adaptation Eq. 27; state dynamics Eq. 31.")
    set_notes(s,
              "Two memory mechanisms are active. Adaptation adds a soft spike-frequency term, which is the model's "
              "replacement for reset and refractory. The state dynamics make the leak per-neuron and the initial state "
              "learnable. Both are cheap: 100 weights for adaptation, 100,000 for z0.")


def slide_mechanisms2(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · MECHANISMS · 2/2", "Delays, two-stage mixing, surrogate gradient")
    xs = [0.7, 4.55, 8.4]
    w = 3.65
    titles = ["LEARNABLE DELAYS", "TWO-STAGE MIXING", "SURROGATE GRADIENT"]
    for x, title in zip(xs, titles):
        add_card(s, x, 1.86, w, 3.55, fill=WHITE)
        add_text(s, x + 0.22, 2.02, w - 0.4, 0.25, title, size=8.5, color=TEAL, bold=True)

    add_math(s, 0.92, 2.42, 3.2, 0.4, "d_c = 20 σ(raw_c)", size=15)
    add_math(s, 0.92, 2.88, 3.2, 0.4, "raw_c ∈ [−10, 10]", size=11)
    add_text(s, 0.92, 3.35, 3.2, 1.0,
             "Per-channel delays in (0, 20) ms, initialised 0.5–10 ms. A fixed two-copy delay line (10 ms stride) "
             "gives the network a short memory before the input projection.",
             size=8.8, color=INK_SOFT, line_spacing=1.1)

    add_math(s, 4.77, 2.42, 3.2, 0.4, "z̃ = tanh(m + λ tanh(m2 + d))", size=12.5)
    add_math(s, 4.77, 2.88, 3.2, 0.4, "λ = 0.1 + 1.9 σ(raw),  init 1", size=10.5)
    add_text(s, 4.77, 3.35, 3.2, 1.0,
             "A second rank-64 path adds a nonlinear correction to the first. λ is a learned scalar that mixes the "
             "two paths; it starts at 1.",
             size=8.8, color=INK_SOFT, line_spacing=1.1)

    add_math(s, 8.62, 2.42, 3.2, 0.4, "forward:  H(o)", size=14)
    add_math(s, 8.62, 2.86, 3.2, 0.4, "backward:  1/(1+2|o|)^2", size=12.5)
    add_text(s, 8.62, 3.35, 3.2, 1.0,
             "The forward spike is exact; the backward pass uses the SLAYER surrogate. The derivative is never "
             "evaluated at a true discontinuity.",
             size=8.8, color=INK_SOFT, line_spacing=1.1)

    add_card(s, 0.7, 5.65, 11.9, 1.05, fill=OCHRE_PALE, line=OCHRE_PALE)
    add_text(s, 0.95, 5.78, 11.35, 0.25, "WHAT IS ACTUALLY IN THE REPORTED RUN", size=8.5, color=OCHRE, bold=True)
    add_text(s, 0.95, 6.03, 11.35, 0.55,
             "The two-stage path and the SLAYER surrogate are active. The adLIF block is configured but inert: its "
             "currents are subtracted from a pre-activation that the two-stage target does not read. Reported model = "
             "adaptation + two-stage mixing.",
             size=9.2, color=INK_SOFT, line_spacing=1.1)
    add_footer(s, 12, source="Methods §2.7.3–2.7.7; adLIF inert in the reported configuration, see §2.5.2.")
    set_notes(s,
              "Three more mechanisms. Per-channel delays and a fixed delay line give the input a short temporal "
              "context. The two-stage mixer adds a second rank-64 path and a learned scalar lambda. The surrogate "
              "gradient is SLAYER: exact Heaviside forward, smooth 1/(1+2|o|)^2 backward. The adLIF block is configured "
              "but does not affect the reported forward pass, which is stated explicitly.")


def slide_augmentation(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · AUGMENTATION", "Only the training events are perturbed")
    add_card(s, 0.7, 1.86, 6.55, 4.45, fill=WHITE)
    add_text(s, 0.95, 2.02, 6.0, 0.25, "SAME-CLASS BLEND, IN ORDER", size=8.8, color=TEAL, bold=True)
    lines = [
        "if A or B is empty: return A",
        "target = (centre(A) + centre(B)) / 2",
        "A', B' = shift each to target",
        "keep each event with p = 0.5",
        "merge, sort by time",
        "keep only 0 ≤ t < W",
    ]
    for i, line in enumerate(lines):
        add_text(s, 0.98, 2.4 + i * 0.42, 6.0, 0.3, line, size=12, color=INK, font="Consolas")
    add_text(s, 0.95, 5.05, 6.0, 0.9,
             "A and B are two raw-event trials of the same class, before binning. The partner is drawn from FIT only. "
             "The output is a new event list, not a weighted average of bins.",
             size=9.2, color=INK_SOFT, line_spacing=1.12)

    add_card(s, 7.55, 1.86, 5.05, 2.35, fill=WHITE)
    add_text(s, 7.78, 2.02, 4.5, 0.25, "ACTIVE IN TRAINING", size=8.8, color=TEAL, bold=True)
    add_text(s, 7.78, 2.35, 4.55, 0.3, "same-class blend  ·  p = 0.5, keep = 0.5", size=9.2, color=INK, bold=True)
    add_text(s, 7.78, 2.72, 4.55, 0.3, "event dropout  ·  p = 0.1", size=9.2, color=INK, bold=True)
    add_text(s, 7.78, 3.12, 4.55, 0.8,
             "Both are applied to raw events; VAL and TEST are byte-identical. No augmentation at test time.",
             size=8.9, color=MUTED, line_spacing=1.1)

    add_card(s, 7.55, 4.42, 5.05, 1.15, fill=OCHRE_PALE, line=OCHRE_PALE)
    add_text(s, 7.78, 4.55, 4.55, 0.25, "WHY THESE TWO", size=8.5, color=OCHRE, bold=True)
    add_text(s, 7.78, 4.8, 4.55, 0.7,
             "Blend creates new same-class event patterns without changing the label; dropout forces the readout to "
             "tolerate missing events.", size=8.8, color=INK_SOFT, line_spacing=1.1)

    add_card(s, 0.7, 6.48, 11.9, 0.0, fill=PAPER, line=PAPER)
    add_text(s, 0.7, 6.46, 11.9, 0.3,
             "Screened and not carried: jitter, insertion, global shift, channel dropout (Results 3.5).",
             size=9, color=MUTED)
    add_footer(s, 13, source="Methods §2.7.9; Algorithm 3; augmentation is train-only, raw-event level.")
    set_notes(s,
              "Augmentation is the only place the training distribution changes. Same-class blend aligns two raw-event "
              "trials by centre of mass, keeps each event independently with p=0.5, merges and clips. Event dropout "
              "deletes each event with p=0.1. Both happen before binning; the validation and test files are untouched. "
              "The other four operations were screened and not carried.")


def slide_training(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · TRAINING PROTOCOL", "Thirty epochs, one seed, checkpointed BPTT")
    boxes = [
        ("SHD events", "7,340 FIT; 816 VAL\n2 ms bins", 0.7, TEAL),
        ("Delay line", "2 copies\n10 ms stride", 2.57, TEAL),
        ("Forward", "10 checkpointed\nchunks of 55", 4.44, TEAL),
        ("Loss", "CE + 1e-4 rate\n10 Hz target", 6.31, OCHRE),
        ("Adam", "1e-3, cosine\n3% warmup", 8.18, OCHRE),
        ("Checkpoint", "10 chunks of 55;\nfinal weights reported", 10.05, BLUE_GRAY),
    ]
    for i, (title, sub, x, acc) in enumerate(boxes):
        flow_box(s, x, 1.9, 1.65, 1.1, title, sub, accent=acc)
        if i < len(boxes) - 1:
            add_line(s, x + 1.68, 2.45, x + 1.84, 2.45, color=MUTED, width=0.8, arrow=True)

    cards = [
        ("LOSS", "cross-entropy + 1e-4 × mean((rate − 10 Hz)²)", "The rate term is a soft penalty on the mean spike rate; it prevents silent or saturated solutions."),
        ("OPTIMISER", "Adam, lr 1e-3, cosine, warmup 3%, clip 1.0", "30 epochs, batch 96; the schedule is step-based, not epoch-based."),
        ("PRECISION", "fp16 state, fp32 masters, GradScaler", "No autocast; explicit casts keep the state dynamics stable."),
        ("CHECKPOINTING", "10 chunks × 55 bins; use_reentrant=False", "State, adaptation and adLIF currents are passed across chunk boundaries."),
    ]
    for i, (lab, big, sub) in enumerate(cards):
        x = 0.7 + (i % 2) * 6.0
        y = 3.35 + (i // 2) * 1.48
        add_card(s, x, y, 5.75, 1.32, fill=WHITE)
        add_text(s, x + 0.2, y + 0.12, 5.3, 0.22, lab, size=8.2, color=TEAL, bold=True)
        add_text(s, x + 0.2, y + 0.36, 5.35, 0.3, big, size=10.5, color=INK, bold=True)
        add_text(s, x + 0.2, y + 0.72, 5.35, 0.5, sub, size=8.2, color=MUTED, line_spacing=1.08)

    add_text(s, 0.7, 6.45, 11.9, 0.3,
             "Single seed (0). Five-epoch screens selected the block set; the 30-epoch run is the reported model.",
             size=8.8, color=MUTED)
    add_footer(s, 14, source="Methods §2.8; 3,069,201 parameters; 5,055 MiB peak; ~505 s/epoch on the RTX 3060.")
    set_notes(s,
              "The training loop is standard BPTT with two non-standard choices: ten checkpointed chunks so the graph "
              "fits in 6 GiB, and a rate penalty that keeps the mean firing rate near 10 Hz. The 30-epoch run uses Adam "
              "with a cosine schedule and a 3% warmup. The reported checkpoint is the final epoch; best-VAL is opt-in.")


def slide_controls_design(prs):
    s = new_slide(prs)
    add_header(s, "METHODS · CONTROLS", "Four controls isolate the vector")
    cards = [
        ("SCALAR, SAME N", "D = 1, N = 64 / 100", "Same neurons, same readout, no state width. The cleanest test of width alone."),
        ("CAPACITY-MATCHED", "D = 1, N = 1,411", "Same parameter budget as the vector model, but a scalar state. Tests whether parameters explain the gap."),
        ("BROADCAST", "D = 1, E = 1000", "Scalar state, 1,000-wide transmission. Separates the width of the state from the width of what a spike sends."),
        ("WIDTH SWEEP", "D = 4 → 1000, N = 64", "Same architecture, five widths. A dose-response curve, not a single comparison."),
    ]
    for i, (title, big, sub) in enumerate(cards):
        x = 0.7 + (i % 2) * 6.0
        y = 1.9 + (i // 2) * 2.15
        add_card(s, x, y, 5.75, 1.95, fill=WHITE)
        add_text(s, x + 0.22, y + 0.16, 5.3, 0.24, title, size=8.5, color=TEAL, bold=True)
        add_text(s, x + 0.22, y + 0.45, 5.3, 0.35, big, size=14, color=INK, bold=True)
        add_text(s, x + 0.22, y + 0.92, 5.3, 0.9, sub, size=9.2, color=INK_SOFT, line_spacing=1.12)
    add_card(s, 0.7, 6.18, 11.9, 0.55, fill=TEAL_DARK, line=TEAL_DARK)
    add_text(s, 0.95, 6.29, 11.35, 0.3,
             "The broadcast control is the key: it can recover transmission width without state width.",
             size=9.5, color=WHITE, bold=True)
    add_footer(s, 15, source="Methods §2.8; control definitions and the width sweep are the design, results follow.")
    set_notes(s,
              "The controls are the argument. A scalar with the same neuron count tests width. A capacity-matched scalar "
              "tests whether parameters alone explain the gap. The broadcast control keeps a scalar state but gives it a "
              "1,000-wide transmission, so it separates state width from transmission width. The width sweep turns the "
              "comparison into a dose-response curve.")


def slide_results_learning(prs):
    s = new_slide(prs)
    add_header(s, "RESULTS · 1/5", "It learns SHD")
    add_figure(s, 0.7, 1.86, 6.85, 4.55, "fig05_training_curves.png",
               "Figure 3a. Fit, validation and official test accuracy over 30 epochs. Best test 88.74%; final 88.07%.")
    compact_stat(s, 7.9, 1.86, 2.3, 1.42, "88.07%", "final TEST", "1994 / 2264", accent=TEAL, value_size=23)
    compact_stat(s, 10.35, 1.86, 2.3, 1.42, "99.78%", "FIT", "first 7,340", accent=BLUE_GRAY, value_size=23)
    compact_stat(s, 7.9, 3.42, 2.3, 1.42, "97.67%", "VAL", "816 trials", accent=BLUE_GRAY, value_size=23)
    compact_stat(s, 10.35, 3.42, 2.3, 1.42, "85.87%", "unseen", "speakers 4 & 5", accent=OCHRE, value_size=23)
    add_text(s, 7.9, 4.95, 4.75, 1.2,
             "It still generalises to unseen speakers: 85.87% on the 1,840 trials from speakers 4 & 5, versus "
             "97.64% on the 424 seen-speaker trials.",
             size=9.2, color=INK_SOFT, line_spacing=1.12)
    add_footer(s, 16, source="Confirmed: N=100, D=1000, 30 epochs, seed 0, 3,069,201 params; official test n=2,264.")
    set_notes(s,
              "The final model reaches 88.07% on the official test set: 1,994 of 2,264. Fit is 99.78%, validation is "
              "97.67%, so the model is not failing to fit. The unseen-speaker split is the closest thing to a "
              "generalisation estimate: 85.87% on the 1,840 trials from speakers 4 and 5, which never appear in training.")


def slide_results_width(prs):
    s = new_slide(prs)
    add_header(s, "RESULTS · 2/5", "Width helps; parameters do not explain it")
    add_figure(s, 0.7, 1.86, 5.85, 4.05, "fig03_width_sweep.png",
               "Figure 2a. TEST accuracy vs state width at N=64, five epochs.")
    add_figure(s, 6.75, 1.86, 5.85, 4.05, "fig04_controls.png",
               "Figure 2b. Scalar, capacity-matched scalar, broadcast, and vector controls at the same readout.")
    add_text(s, 0.7, 6.08, 11.9, 0.55,
             "D=4→256 buys +20.4 pp; D=1000 is not better than D=256 at this budget. At N=100, five epochs: "
             "scalar 35.91%, broadcast 65.42%, vector 79.64%. A 2M-parameter scalar reaches 33.13%.",
             size=9.4, color=INK_SOFT, line_spacing=1.12)
    add_footer(s, 17, source="Screening (5 epochs) except where noted; all numbers on the official test set, n=2,264.")
    set_notes(s,
              "Left: width sweep at fixed N=64. Accuracy rises from 57.20% at D=4 to 77.61% at D=256, then flattens. "
              "Right: the controls. The scalar at N=64 is 49.43% at 20 epochs; the capacity-matched 1,411-neuron scalar "
              "is 33.13% at five epochs. The broadcast control recovers a large part of the gap with a scalar state, "
              "which is exactly why it is the sharpest control.")


def slide_results_scalar_sanity(prs):
    s = new_slide(prs)
    add_header(s, "RESULTS · 3/5", "A 1,000-neuron scalar still does not fit SHD",
               "More neurons at D=1 is not the same axis as more state width.")

    # Left: the sanity-check scalar.
    add_card(s, 0.7, 1.88, 5.85, 2.72, fill=WHITE)
    add_rect(s, 0.7, 1.88, 0.06, 2.72, RED)
    add_text(s, 0.95, 2.04, 5.3, 0.22, "SCALAR  ·  N=1,000  ·  D=1", size=9.5, color=RED, bold=True)
    add_text(s, 0.95, 2.30, 5.3, 0.22, "30 EPOCHS  ·  1,024,703 PARAMS", size=8.2, color=MUTED)
    add_text(s, 0.95, 2.58, 5.3, 0.72, "43.6%", size=40, color=INK, bold=True)
    add_text(s, 0.95, 3.30, 5.3, 0.24, "TEST  ·  987 / 2,264", size=10.5, color=INK_SOFT)
    add_text(s, 0.95, 3.62, 5.3, 0.24, "FIT 49.6%   ·   VAL 42.7%", size=10.5, color=INK_SOFT)
    add_text(s, 0.95, 4.02, 5.3, 0.24, "34 min wall-clock  ·  67 s / epoch", size=8.5, color=MUTED)

    # Right: the five-epoch vector it is compared with.
    add_card(s, 6.78, 1.88, 5.85, 2.72, fill=WHITE)
    add_rect(s, 6.78, 1.88, 0.06, 2.72, TEAL)
    add_text(s, 7.03, 2.04, 5.3, 0.22, "VECTOR  ·  N=100  ·  D=1,000", size=9.5, color=TEAL, bold=True)
    add_text(s, 7.03, 2.30, 5.3, 0.22, "5 EPOCHS  ·  2,140,100 PARAMS", size=8.2, color=MUTED)
    add_text(s, 7.03, 2.58, 5.3, 0.72, "76.7%", size=40, color=INK, bold=True)
    add_text(s, 7.03, 3.30, 5.3, 0.24, "TEST  ·  1,736 / 2,264", size=10.5, color=INK_SOFT)
    add_text(s, 7.03, 3.62, 5.3, 0.24, "FIT 96.4%   ·   VAL 93.5%", size=10.5, color=INK_SOFT)
    add_text(s, 7.03, 4.02, 5.3, 0.24, "31 min wall-clock  ·  343 s / epoch", size=8.5, color=MUTED)

    add_card(s, 0.7, 4.92, 11.9, 1.50, fill=TEAL_DARK, line=TEAL_DARK)
    add_text(s, 0.95, 5.04, 11.3, 0.22, "THE TELL", size=8.5, color=RGBColor(0xA9, 0xD3, 0xD2), bold=True)
    add_text(s, 0.95, 5.28, 11.3, 1.02,
             "49.6% on FIT: the 1,000-neuron scalar does not fit the training split, so this is not a "
             "generalisation gap. In the same half-hour the vector gets 5 passes instead of 30 and still "
             "gains 33 points of TEST. The 30-epoch vector reaches 88.1% TEST and 99.8% FIT; the scalar "
             "never gets above 50% on FIT.",
             size=10.8, color=WHITE, line_spacing=1.14)
    add_text(s, 0.95, 5.94, 11.3, 0.28,
             "The variable that moves accuracy is state width, not neuron count.",
             size=11.2, color=RGBColor(0xA9, 0xD3, 0xD2), bold=True)

    add_text(s, 0.7, 6.50, 11.9, 0.24,
             "Not parameter-matched: 1.02M scalar vs 2.14M vector; the scalar spends 1M weights on a full recurrent matrix.",
             size=7.6, color=MUTED)
    add_footer(s, 18, source="Scalar: v3_emerg_n1000_d1_ep30 (30 ep); vector: 2026-10-26_20-55-54 (5 ep); final: v3_n100_spec_ep30.")
    set_notes(s,
              "Sanity check requested after the draft: N=1,000 at D=1, 30 epochs, reaches 43.6% TEST and only 49.6% FIT, "
              "in 34 min. The five-epoch N=100 D=1000 run reaches 76.7% TEST and 96.4% FIT in 31 min. This is not a "
              "like-for-like parameter comparison, 1.02M vs 2.14M, but it rules out the simple 'more neurons' reading: "
              "the scalar has 1M recurrent weights and still cannot fit. The 30-epoch vector is 88.07% TEST / 99.78% FIT. "
              "Caveat: the five-epoch vector is the earlier 2.14M configuration, not the final 3.07M one.")


def slide_results_state(prs):
    s = new_slide(prs)
    add_header(s, "RESULTS · 4/5", "What the state carries")
    add_figure(s, 0.7, 1.86, 3.85, 3.35, "fig08_probe.png",
               "Figure 4a. Linear probes on frozen states.")
    add_figure(s, 4.72, 1.86, 3.85, 3.35, "fig09_effective_dim.png",
               "Figure 4b. Effective dimension: 7.49 of 100,000.")
    add_figure(s, 8.74, 1.86, 3.85, 3.35, "fig10_frozen_dynamics.png",
               "Figure 4c. Learned vs frozen dynamics.")
    add_text(s, 0.7, 5.45, 11.9, 1.1,
             "Spike counts decode slightly better than the trial-mean state (87.81% vs 84.67%); the final state alone is "
             "27.61%, so the code is in the trajectory, not the endpoint. The population mean uses 7.49 effective "
             "dimensions, and learned dynamics are worth about 40 pp over an untrained reservoir.",
             size=9.4, color=INK_SOFT, line_spacing=1.18)
    add_footer(s, 19, source="Frozen final checkpoint; probes on FIT; reported as accuracy on the 2,264 test trials.")
    set_notes(s,
              "The probes ask what the state contains. Spike counts decode 87.81%, the trial-mean state 84.67%, the final "
              "state only 27.61%: the information is in the trajectory. The participation ratio is 7.49 effective "
              "dimensions out of 100,000, and nine components reach 90% of the variance. The right panel shows that the "
              "learned dynamics are worth about 40 pp over a shuffled or frozen reservoir.")


def slide_results_code(prs):
    s = new_slide(prs)
    add_header(s, "RESULTS · 5/5", "A distributed, redundant code")
    add_figure(s, 0.7, 1.86, 3.85, 3.35, "fig11_mi_distribution.png",
               "Figure 5a. MI per neuron: median 0.502 bits.")
    add_figure(s, 4.72, 1.86, 3.85, 3.35, "fig12_knockout_drop.png",
               "Figure 5b. Single silencing: 0.30 ± 0.63 pp.")
    add_figure(s, 8.74, 1.86, 3.85, 3.35, "fig13_cumulative_topk.png",
               "Figure 5c. Cumulative top-k: smooth decay.")
    add_text(s, 0.7, 5.45, 11.9, 1.1,
             "Information is spread across the population: median mutual information 0.502 bits, 91 of 100 neurons "
             "informative, and silencing one neuron costs only 0.30 pp on average. The top-k curve degrades smoothly, "
             "which is what a distributed code predicts and a dedicated detector does not.",
             size=9.4, color=INK_SOFT, line_spacing=1.18)
    add_footer(s, 20, source="FIT mutual information; knockouts on the official test set; N=100, 30 epochs.")
    set_notes(s,
              "The code is distributed. Every neuron carries some class information, the median is about half a bit, "
              "and no single neuron is necessary: silencing one costs 0.30 pp on average, with some deletions even "
              "improving accuracy. The cumulative top-k curve falls smoothly, which is the signature of a redundant "
              "population code rather than a sparse, dedicated one.")


def slide_future(prs):
    s = new_slide(prs)
    add_header(s, "FUTURE DIRECTIONS", "Make it cheaper; give each neuron an identity")

    add_card(s, 0.7, 1.86, 5.85, 4.3, fill=WHITE)
    add_text(s, 0.95, 2.02, 5.3, 0.25, "REDUCE THE FOOTPRINT", size=8.8, color=TEAL, bold=True)
    add_text(s, 0.95, 2.38, 5.3, 0.9,
             "91% of the parameters sit in two dense 1,000-wide maps: the input projection (1.4M) and the "
             "emission matrix (1M).", size=10.2, color=INK_SOFT, line_spacing=1.14)
    add_math(s, 0.95, 3.34, 5.3, 0.4, "W_emit ≈ D + U Vᵀ", size=17)
    add_text(s, 0.95, 3.82, 5.3, 1.25,
             "Diagonalise the dense emitter and keep a rank-r correction. The cost becomes 1,000 + 2,000r; "
             "at r = 16 that is about 33k parameters, not 1M. The same factorization applies to the input matrix.",
             size=9.4, color=INK_SOFT, line_spacing=1.13)
    add_chip(s, 0.95, 5.22, 1.55, 0.3, "1M → ~33k", fill=TEAL_PALE, color=TEAL_DARK, size=8)
    add_chip(s, 2.65, 5.22, 2.1, 0.3, "input: 1.4M → ~38k", fill=TEAL_PALE, color=TEAL_DARK, size=8)

    add_card(s, 6.75, 1.86, 5.85, 4.3, fill=WHITE)
    add_text(s, 7.0, 2.02, 5.3, 0.25, "PER-NEURON EMBEDDING", size=8.8, color=OCHRE, bold=True)
    add_math(s, 7.0, 2.38, 5.3, 0.4, "e_i ∈ ℝ^d,   g_i = W_g e_i,   β_i = W_β e_i", size=13)
    add_text(s, 7.0, 2.92, 5.3, 1.35,
             "Share the 1,000-D map and learn only a small embedding per neuron. At d = 8, one 1,000-D head "
             "costs 8.8k instead of 100k; the same embedding can drive the gate, the bias and the readout.",
             size=9.4, color=INK_SOFT, line_spacing=1.13)
    add_math(s, 7.0, 4.38, 5.3, 0.4, "w_out,i = W_o e_i", size=14)
    add_text(s, 7.0, 4.82, 5.3, 0.8,
             "The question is whether a neuron needs a full private vector, or just a compact identity that "
             "selects a shared pattern.", size=9.4, color=INK_SOFT, line_spacing=1.13)
    add_chip(s, 7.0, 5.22, 2.0, 0.3, "d = 8: 8.8k / head", fill=OCHRE_PALE, color=OCHRE, size=8)

    add_card(s, 0.7, 6.3, 11.9, 0.55, fill=TEAL_PALE, line=TEAL_PALE)
    add_text(s, 0.95, 6.41, 11.35, 0.3,
             "Both are design sketches, not measured results. The test is whether the SHD gain survives at a much smaller budget.",
             size=9.4, color=INK_SOFT)
    add_footer(s, 21, source="Future work; no results claimed. Diagonal + low-rank and embedding costs are parameter counts, not run outcomes.")
    set_notes(s,
              "Two concrete directions. First, shrink the footprint: the dense input and emission matrices dominate the "
              "parameter budget, so diagonalising the emitter and keeping a low-rank correction, and factorising the input "
              "projection, could remove most of that cost. Second, replace the 100 independent 1,000-D gates and biases "
              "with a small learned embedding per neuron and a shared map; the embedding can also condition the readout. "
              "Neither is measured yet — they are the next experiments, and the criterion is whether SHD accuracy survives.")


def slide_close(prs):
    s = new_slide(prs)
    add_header(s, "WHAT WE CAN AND CANNOT CLAIM", "Takeaways")
    add_card(s, 0.7, 1.86, 5.75, 3.05, fill=WHITE)
    add_text(s, 0.95, 2.02, 5.2, 0.25, "SHOWN", size=8.8, color=TEAL, bold=True)
    bullets = [
        "A vector state beats a scalar state at fixed N and readout.",
        "Width has a dose-response: D=4→256 buys +20.4 pp.",
        "The gap is not parameter count: 2M scalar 33.13%.",
        "The code is distributed: no critical unit.",
    ]
    for i, b in enumerate(bullets):
        add_text(s, 0.95, 2.4 + i * 0.56, 5.2, 0.45, "—  " + b, size=9.4, color=INK_SOFT, line_spacing=1.1)

    add_card(s, 6.6, 1.86, 6.0, 3.05, fill=WHITE)
    add_text(s, 6.85, 2.02, 5.5, 0.25, "NOT YET", size=8.8, color=OCHRE, bold=True)
    lim = [
        "Single seed; no seed variance.",
        "Blocks selected on five-epoch TEST.",
        "No N=100 all-off baseline; scalar controls untuned.",
        "VAL is speaker-overlapping; held-out pilot overstates.",
    ]
    for i, b in enumerate(lim):
        add_text(s, 6.85, 2.4 + i * 0.56, 5.4, 0.45, "—  " + b, size=9.4, color=INK_SOFT, line_spacing=1.1)

    add_card(s, 0.7, 5.2, 11.9, 1.15, fill=TEAL_DARK, line=TEAL_DARK)
    add_text(s, 0.95, 5.34, 11.35, 0.25, "THE CLAIM", size=8.5, color=RGBColor(0xA9, 0xD3, 0xD2), bold=True)
    add_text(s, 0.95, 5.6, 11.35, 0.6,
             "On SHD, the width of a neuron's internal state — not its parameter count — is the variable that buys "
             "computation. The mechanism is a spike that transmits the whole state, not a wider scalar.",
             size=11.2, color=WHITE, line_spacing=1.15)
    add_text(s, 0.7, 6.55, 11.9, 0.3,
             "REPO  ·  [ paste repository URL here ]", size=9, color=MUTED)
    add_footer(s, 22, source="All numbers from the current Results draft; confirmed vs screening tags in the manuscript.")
    set_notes(s,
              "The claim is deliberately narrow. The model shows that state width buys accuracy at fixed neuron count "
              "and readout, that the effect is not parameter count, and that the code is distributed. The limitations "
              "are single seed, five-epoch selection, an unmatched N=100 baseline, and a speaker-overlapping validation "
              "split. Those are the next experiments.")


def build():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    for fn in [
        slide_title, slide_question, slide_shd, slide_architecture,
        slide_input, slide_mixing, slide_broadcast, slide_state,
        slide_forward, slide_init, slide_mechanisms1, slide_mechanisms2,
        slide_augmentation, slide_training, slide_controls_design,
        slide_results_learning, slide_results_width, slide_results_scalar_sanity,
        slide_results_state, slide_results_code, slide_future, slide_close,
    ]:
        fn(prs)
    prs.save(OUT)
    print(f"saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
