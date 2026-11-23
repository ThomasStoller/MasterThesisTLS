"""Generate fig_feature_map.pdf.

The figure maps the 42 BLE features onto one advertising PDU and onto the
capture timeline. It is drawn entirely with matplotlib primitives, so the
output is a vector PDF with selectable text and no external images.

Usage:
    python make_fig_feature_map.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pathlib

# ----------------------------------------------------------------------------
# palette and typography
# ----------------------------------------------------------------------------
TEXT = "#1f2933"
GREY = "#5f6b7a"
BORDER = "#b8c4d0"
FILL_BLUE = "#e8eef7"
FILL_PINK = "#f8e8e8"
FILL_GREEN = "#eaf3ea"
FILL_BOX = "#f7f9fc"
FILL_TOP = "#f5f7fa"
FILL_AD = "#f3f8f3"
ACCENT = "#4a7fb5"

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

W, H = 12.6, 7.6

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)


def txt(x, y, s, size=9, weight="normal", color=TEXT, family=SANS,
        ha="left", va="center", style="normal", ls=1.1):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            family=family, ha=ha, va=va, style=style, linespacing=ls, zorder=5)


def rbox(x, y, w, h, fc, ec=BORDER, lw=0.8, ls="-", rounding=0.07, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def rect(x, y, w, h, fc, ec=BORDER, lw=0.8, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8,
                                shrinkA=0, shrinkB=0), zorder=3)


def dashed(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#9aa8b8", lw=0.7,
            ls=(0, (2.5, 2.5)), zorder=1)


# ----------------------------------------------------------------------------
# capture timeline (top left)
# ----------------------------------------------------------------------------
txt(0.24, 7.40, "capture timeline", 11, "bold")
txt(0.24, 7.19, "several packets from one source, and what else transmits nearby",
    8.5, color=GREY, style="italic")
txt(1.01, 7.04, "500 ms bin", 8.5, color=GREY, ha="center")
ax.plot([0.35, 0.35, 1.69, 1.69], [6.88, 6.94, 6.94, 6.88], color=GREY, lw=0.8)

box_x = [0.35, 0.85, 1.38, 1.90, 2.42, 2.94]
box_w, box_h, box_y = 0.25, 0.38, 6.38
for x in box_x:
    rect(x, box_y, box_w, box_h, FILL_BLUE)
for a, b in zip(box_x[:-1], box_x[1:]):
    arrow(a + box_w + 0.02, b - 0.02, box_y + box_h / 2)
txt(0.72, 6.16, "IAT", 8.5, color=GREY, ha="center")

# ----------------------------------------------------------------------------
# timing and joint activity (top right)
# ----------------------------------------------------------------------------
rbox(3.57, 5.93, 7.85, 1.10, FILL_TOP, rounding=0.08)
txt(3.88, 6.85, "Timing and joint activity  (16 features)", 11.5, "bold")
timing_lines = [
    "iat_median, iat_mean, iat_std, iat_deviation, is_burst_iat,",
    "packets_within_2ms / 25ms / 100ms / 1s,",
    "bin_total_packets, bin_distinct_sources, coactive_packets_in_bin,",
    "coactive_sources_in_bin, coactive_density, is_solo_bin, bin_sync_flips",
]
y = 6.60
for line in timing_lines:
    txt(3.88, y, line, 8.3, family=MONO)
    y -= 0.145

# ----------------------------------------------------------------------------
# advertising PDU (middle)
# ----------------------------------------------------------------------------
txt(0.24, 5.42, "one advertising PDU (legacy, 1 Mbit/s)", 11, "bold")
txt(4.29, 5.42, "advertising channel 37 / 38 / 39  (5 features)",
    9.5, color=ACCENT)

rect(0.316, 5.16, 10.49, 0.10, "none", ec=ACCENT, lw=1.1)

fields = [
    (0.316, 0.884, "Preamble", "1 B", FILL_BLUE),
    (1.200, 1.673, "Access\nAddress", "4 B", FILL_BLUE),
    (2.873, 1.370, "PDU\nheader", "2 B", FILL_BLUE),
    (4.243, 2.070, "AdvA\n(sender address)", "6 B", FILL_PINK),
    (6.313, 3.504, "AdvData", "0 to 31 B", FILL_GREEN),
    (9.817, 0.992, "CRC", "3 B", FILL_BLUE),
]
for x, w, name, size, fc in fields:
    rect(x, 4.22, w, 0.92, fc)
    txt(x + w / 2, 4.87, name, 9.5, color=TEXT, ha="center", va="center")
    txt(x + w / 2, 4.50, size, 8.8, color=GREY, ha="center", va="center")

txt(10.85, 5.10, "is_37, is_38, is_39,", 6.8, color=GREY, family=MONO)
txt(10.85, 4.97, "is_adv_channel, is_broadcast", 6.8, color=GREY, family=MONO)

# ----------------------------------------------------------------------------
# AD structures inside AdvData
# ----------------------------------------------------------------------------
rect(6.30, 3.55, 3.46, 0.56, "none", ec="#8d9aa8", lw=0.8, ls=(0, (3, 3)))
ad_cells = [
    (6.31, 0.76, "Length", "1 B"),
    (7.07, 1.20, "AD Type", "1 to 2 B"),
    (8.27, 1.48, "AD Data", "0 to 29 B"),
]
for x, w, name, size in ad_cells:
    rect(x, 3.56, w, 0.54, FILL_AD)
    txt(x + w / 2, 3.94, name, 8.8, "bold", ha="center", va="center")
    txt(x + w / 2, 3.74, size, 8.3, color=GREY, ha="center", va="center")
txt(8.02, 3.38, "AD structures inside AdvData", 8.5, color=GREY, ha="center")

# ----------------------------------------------------------------------------
# connectors from the PDU to the feature boxes
# ----------------------------------------------------------------------------
dashed(1.64, 4.22, 1.52, 3.02)   # Access Address -> packet geometry
dashed(3.57, 4.22, 4.10, 3.02)   # PDU header -> PDU role
dashed(5.27, 4.22, 6.50, 3.02)   # AdvA -> identity
dashed(9.75, 3.55, 9.98, 3.02)   # AD structures -> payload
dashed(10.30, 4.22, 9.98, 3.02)  # CRC -> payload

# ----------------------------------------------------------------------------
# feature family boxes (bottom)
# ----------------------------------------------------------------------------
BOX_Y, BOX_H = 1.07, 1.95


def family_box(x, w, title, lines, mono=True, first_dark=False):
    rbox(x, BOX_Y, w, BOX_H, FILL_BOX, rounding=0.08)
    txt(x + 0.25, 2.76, title, 11.5, "bold")
    y = 2.42
    for i, line in enumerate(lines):
        color = TEXT if (not first_dark or i == 0) else GREY
        txt(x + 0.25, y, line, 8.3, color=color,
            family=MONO if mono else SANS)
        y -= 0.16


family_box(0.24, 2.56, "Packet geometry  (9)", [
    "len_ad_type_binned,",
    "len_data_binned,",
    "data_len_ratio_binned,",
    "ad_type_len_ratio_binned,",
    "time_to_next_binned,",
    "packet_send_freq_binned,",
    "time_start_end_binned,",
    "delta_end_start_binned,",
    "delta_start_start_binned",
])

family_box(2.94, 2.40, "PDU role  (6)", [
    "is_ADV_IND,",
    "is_ADV_NONCONN_IND,",
    "is_ADV_SCAN_IND,",
    "is_ADV_EXT_IND,",
    "is_AUX_ADV_IND,",
    "is_SCAN_REQ",
])

family_box(5.47, 2.14, "Not used: identity", [
    "source address",
    "(rotates; kept only for",
    "grouped splits, never as",
    "a model feature)",
], mono=False, first_dark=True)

family_box(7.75, 3.68, "Payload shape and dynamics  (6)", [
    "has_company_id, has_uuid16,",
    "service_data_len_ratio,",
    "payload_changed, payload_sync_in_bin,",
    "payload_sync_density",
])

# ----------------------------------------------------------------------------
# footer
# ----------------------------------------------------------------------------
txt(0.24, 0.52,
    "42 features in total. Every feature is computed per packet or per source within a single capture;",
    8.5, color=GREY)
txt(0.24, 0.36,
    "none of them uses the group label, and RSSI is not among them.",
    8.5, color=GREY)

# ----------------------------------------------------------------------------
# save
# ----------------------------------------------------------------------------
out_dir = pathlib.Path(__file__).resolve().parent
fig.savefig(out_dir / "fig_feature_map.pdf", format="pdf")
print("wrote", out_dir / "fig_feature_map.pdf")
