"""generate_pitch_deck.py — BanditDB investor pitch deck generator.

Run:
    python docs/generate_pitch_deck.py
Output:
    docs/BanditDB_Pitch_Deck.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import os

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BanditDB_Pitch_Deck.pptx")

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY     = RGBColor(0x0A, 0x0F, 0x1E)   # slide background
BLUE     = RGBColor(0x21, 0x96, 0xF3)   # accent / highlight
ORANGE   = RGBColor(0xFF, 0x6D, 0x00)   # secondary accent
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
GREY     = RGBColor(0xA0, 0xAA, 0xBB)
DARKCARD = RGBColor(0x0E, 0x1A, 0x2E)   # card / box background

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)

def bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def txbox(slide, left, top, width, height):
    return slide.shapes.add_textbox(left, top, width, height)

def para(tf, text, size, bold=False, color=WHITE, align=PP_ALIGN.LEFT, space_before=0):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return p

def rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def accent_line(slide, left, top, width):
    line = slide.shapes.add_shape(1, left, top, width, Pt(3))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()

# ── Slide 1: Hero ─────────────────────────────────────────────────────────────
def slide_hero(prs):
    s = blank_slide(prs)
    bg(s)

    # Subtle grid texture suggestion — two faint lines
    rect(s, 0, Inches(3.6), W, Pt(1), RGBColor(0x15, 0x25, 0x40))

    # Tag line above logo
    tb = txbox(s, Inches(1.5), Inches(1.6), Inches(10), Inches(0.6))
    tf = tb.text_frame
    tf.word_wrap = False
    para(tf, "INTRODUCING", 11, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    # Main title
    tb = txbox(s, Inches(0.5), Inches(2.1), Inches(12.3), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = False
    para(tf, "BanditDB", 72, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    tb = txbox(s, Inches(1.0), Inches(3.4), Inches(11.3), Inches(0.7))
    tf = tb.text_frame
    para(tf, "The Decision Intelligence Database", 28, color=BLUE, align=PP_ALIGN.CENTER)

    # One-liner
    tb = txbox(s, Inches(2.0), Inches(4.2), Inches(9.3), Inches(0.6))
    tf = tb.text_frame
    para(tf, "Two API calls. Continuous learning. No ML team required.", 16, color=GREY, align=PP_ALIGN.CENTER)

    # Bottom bar
    rect(s, 0, Inches(6.9), W, Inches(0.6), DARKCARD)
    tb = txbox(s, Inches(0.5), Inches(6.95), Inches(12.3), Inches(0.45))
    tf = tb.text_frame
    para(tf, "dynamicpricing.ai  ·  github.com/dynamicpricing-ai/banditdb  ·  Confidential", 9, color=GREY, align=PP_ALIGN.CENTER)

# ── Slide 2: Problem ──────────────────────────────────────────────────────────
def slide_problem(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.25), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.5), Inches(11), Inches(0.8))
    tf = tb.text_frame
    para(tf, "Every app makes decisions.", 36, bold=True, color=WHITE)

    tb = txbox(s, Inches(0.7), Inches(1.4), Inches(11), Inches(0.5))
    tf = tb.text_frame
    para(tf, "Most of them never learn.", 36, bold=True, color=BLUE)

    # Three pain cards
    card_data = [
        ("ML pipelines\ntake months",
         "Data science team, feature stores, model training, A/B infra — 6–18 months before you see ROI."),
        ("A/B tests are\ntoo slow",
         "Split traffic 50/50 for weeks to reach significance. Revenue lost to the losing variant every day."),
        ("Personalization\nis expensive",
         "Real-time recommendation engines cost $500K+ to build and $100K+/year to operate at scale."),
    ]
    for i, (title, body) in enumerate(card_data):
        x = Inches(0.55 + i * 4.2)
        rect(s, x, Inches(2.3), Inches(3.9), Inches(3.9), DARKCARD)
        # Orange top accent
        rect(s, x, Inches(2.3), Inches(3.9), Pt(4), ORANGE)
        tb = txbox(s, x + Inches(0.2), Inches(2.5), Inches(3.5), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, title, 20, bold=True, color=WHITE)
        tb = txbox(s, x + Inches(0.2), Inches(3.55), Inches(3.5), Inches(2.3))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, body, 13, color=GREY)

# ── Slide 3: Solution ─────────────────────────────────────────────────────────
def slide_solution(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.25), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.4), Inches(11), Inches(0.8))
    tf = tb.text_frame
    para(tf, "BanditDB learns from every decision —", 32, bold=True, color=WHITE)

    tb = txbox(s, Inches(0.7), Inches(1.1), Inches(11), Inches(0.6))
    tf = tb.text_frame
    para(tf, "automatically, in real-time, from day one.", 32, bold=True, color=BLUE)

    # Two code boxes side by side
    for i, (label, code, sub) in enumerate([
        ("1. Ask for a decision",
         'POST /predict\n{"campaign_id": "churn",\n "context": [0.8, 0.2, ...]}',
         "Returns the best intervention for this customer"),
        ("2. Tell it what happened",
         'POST /reward\n{"interaction_id": "...",\n "reward": 0.92}',
         "Model updates in microseconds. Learns instantly."),
    ]):
        x = Inches(0.6 + i * 6.4)
        rect(s, x, Inches(2.1), Inches(6.0), Inches(3.9), DARKCARD)
        rect(s, x, Inches(2.1), Inches(6.0), Pt(4), BLUE)

        tb = txbox(s, x + Inches(0.25), Inches(2.25), Inches(5.5), Inches(0.45))
        tf = tb.text_frame
        para(tf, label, 13, bold=True, color=BLUE)

        tb = txbox(s, x + Inches(0.25), Inches(2.75), Inches(5.5), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, code, 14, color=WHITE)
        for p in tb.text_frame.paragraphs:
            for r in p.runs:
                r.font.name = "Courier New"

        tb = txbox(s, x + Inches(0.25), Inches(5.1), Inches(5.5), Inches(0.6))
        tf = tb.text_frame
        para(tf, sub, 11, color=GREY)

    tb = txbox(s, Inches(0.7), Inches(6.3), Inches(11.9), Inches(0.5))
    tf = tb.text_frame
    para(tf, "That's it.  No model training.  No ML team.  No infrastructure.  Ship in an afternoon.", 13, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

# ── Slide 4: How it works ─────────────────────────────────────────────────────
def slide_how(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.3), Inches(11), Inches(0.8))
    tf = tb.text_frame
    para(tf, "Five algorithms. One binary. Autonomous optimization.", 30, bold=True, color=WHITE)

    algo_data = [
        ("LinUCB", "Linear rewards\nFastest convergence\nProduction default"),
        ("Thompson\nSampling", "Bayesian exploration\nBest under\nstochastic rewards"),
        ("NeuralLinUCB", "Non-linear rewards\nMLP embedding\n14× less memory"),
        ("NeuralTS", "Best long-run\nconvergence\nZhang et al. 2021"),
        ("Progressive\nTournament", "Unsure which fits?\nBanditDB picks\nthe winner for you"),
    ]
    for i, (name, desc) in enumerate(algo_data):
        x = Inches(0.4 + i * 2.55)
        rect(s, x, Inches(1.65), Inches(2.3), Inches(4.0), DARKCARD)
        rect(s, x, Inches(1.65), Inches(2.3), Pt(4), BLUE if i < 4 else ORANGE)
        tb = txbox(s, x + Inches(0.15), Inches(1.85), Inches(2.0), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, name, 15, bold=True, color=WHITE)
        tb = txbox(s, x + Inches(0.15), Inches(2.75), Inches(2.0), Inches(2.5))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, desc, 12, color=GREY)

    tb = txbox(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(0.5))
    tf = tb.text_frame
    para(tf, "Written in Rust · Sub-millisecond predictions · WAL-based durability · Runs on a $5/mo VPS", 12, color=GREY, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.7), Inches(6.4), Inches(11.9), Inches(0.6))
    tf = tb.text_frame
    para(tf, "Works natively with Claude, GPT, and other AI agents via MCP (Model Context Protocol)", 13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

# ── Slide 5: Traction / Numbers ───────────────────────────────────────────────
def slide_traction(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.3), Inches(11), Inches(0.85))
    tf = tb.text_frame
    para(tf, "Real results. Measured against ground truth.", 30, bold=True, color=WHITE)

    stats = [
        ("91%", "of oracle performance\non real customer data"),
        ("75.6%", "normalized improvement\nvs random baseline"),
        ("<1ms", "prediction latency\nat 1,000+ req/sec"),
        ("21×", "less memory than\nnaive ML approach"),
        ("2 calls", "to integrate\npredict + reward"),
    ]
    for i, (big, small) in enumerate(stats):
        x = Inches(0.45 + i * 2.55)
        rect(s, x, Inches(1.6), Inches(2.35), Inches(3.5), DARKCARD)
        rect(s, x, Inches(1.6), Inches(2.35), Pt(4), BLUE)
        tb = txbox(s, x + Inches(0.1), Inches(1.8), Inches(2.15), Inches(1.1))
        tf = tb.text_frame
        para(tf, big, 38, bold=True, color=BLUE)
        tb = txbox(s, x + Inches(0.1), Inches(2.95), Inches(2.15), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, small, 12, color=GREY)

    # Source note
    tb = txbox(s, Inches(0.7), Inches(5.35), Inches(11.9), Inches(0.4))
    tf = tb.text_frame
    para(tf, "B2B SaaS churn prevention benchmark · NeuralLinUCB · 384-dim sentence embeddings · 20K interactions", 10, color=GREY, align=PP_ALIGN.CENTER)

    # Use case strip
    rect(s, 0, Inches(5.9), W, Inches(1.35), DARKCARD)
    tb = txbox(s, Inches(0.7), Inches(6.0), Inches(11.9), Inches(0.4))
    tf = tb.text_frame
    para(tf, "PROVEN USE CASES", 10, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    tb = txbox(s, Inches(0.7), Inches(6.4), Inches(11.9), Inches(0.7))
    tf = tb.text_frame
    para(tf, "Churn prevention  ·  Dynamic pricing  ·  Content recommendation  ·  A/B testing  ·  Clinical trials  ·  AI agent decisions", 14, color=WHITE, align=PP_ALIGN.CENTER)

# ── Slide 6: Market ───────────────────────────────────────────────────────────
def slide_market(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.3), Inches(11), Inches(0.85))
    tf = tb.text_frame
    para(tf, "Every app that makes decisions is a customer.", 30, bold=True, color=WHITE)

    market_data = [
        ("$47B", "ML/AI\nInfrastructure\nMarket (2026)", BLUE),
        ("$12B", "Personalization\nEngine\nMarket (2026)", BLUE),
        ("$8B", "Decision\nIntelligence\nMarket (2026)", ORANGE),
    ]
    for i, (size, label, color) in enumerate(market_data):
        x = Inches(1.0 + i * 3.8)
        rect(s, x, Inches(1.7), Inches(3.3), Inches(3.5), DARKCARD)
        tb = txbox(s, x + Inches(0.2), Inches(2.0), Inches(2.9), Inches(1.3))
        tf = tb.text_frame
        para(tf, size, 48, bold=True, color=color)
        tb = txbox(s, x + Inches(0.2), Inches(3.35), Inches(2.9), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, label, 13, color=GREY)

    tb = txbox(s, Inches(0.7), Inches(5.5), Inches(11.9), Inches(0.5))
    tf = tb.text_frame
    para(tf, "BanditDB sits at the intersection — a new category:", 16, color=GREY, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.7), Inches(6.0), Inches(11.9), Inches(0.7))
    tf = tb.text_frame
    para(tf, "Decision Intelligence Infrastructure", 24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.7), Inches(6.65), Inches(11.9), Inches(0.5))
    tf = tb.text_frame
    para(tf, "The database layer that every AI-native application will need.", 14, color=BLUE, align=PP_ALIGN.CENTER)

# ── Slide 7: Why Now ──────────────────────────────────────────────────────────
def slide_why_now(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.3), Inches(11), Inches(0.85))
    tf = tb.text_frame
    para(tf, "AI agents need decision infrastructure.", 30, bold=True, color=WHITE)

    reasons = [
        ("AI agents make\nmillions of decisions",
         "GPT, Claude, and Gemini agents take actions at scale. None of them currently learn which decisions work best. BanditDB closes that loop."),
        ("The data science\nbottleneck is real",
         "73% of ML projects never reach production (Gartner). BanditDB ships learning in an afternoon without a data scientist."),
        ("Rust + open source\nis the right moat",
         "Sub-millisecond latency, 21× memory efficiency, Apache 2.0. The infrastructure layer wins by being fast, cheap, and trusted."),
    ]
    for i, (title, body) in enumerate(reasons):
        y = Inches(1.65 + i * 1.75)
        rect(s, Inches(0.6), y, Inches(12.1), Inches(1.5), DARKCARD)
        rect(s, Inches(0.6), y, Pt(4), Inches(1.5), ORANGE)

        tb = txbox(s, Inches(0.9), y + Inches(0.15), Inches(3.0), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, title, 16, bold=True, color=WHITE)

        tb = txbox(s, Inches(3.9), y + Inches(0.15), Inches(8.5), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, body, 13, color=GREY)

# ── Slide 8: Ask ──────────────────────────────────────────────────────────────
def slide_ask(prs):
    s = blank_slide(prs)
    bg(s)

    # Full-width top bar
    rect(s, 0, 0, W, Inches(0.8), BLUE)
    tb = txbox(s, Inches(0.5), Inches(0.1), Inches(12.3), Inches(0.6))
    tf = tb.text_frame
    para(tf, "BanditDB  ·  Seed Round", 18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.7), Inches(1.0), Inches(11.9), Inches(0.9))
    tf = tb.text_frame
    para(tf, "We're raising $2M to make BanditDB", 34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.7), Inches(1.8), Inches(11.9), Inches(0.7))
    tf = tb.text_frame
    para(tf, "the decision layer for every AI-native application.", 34, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    use_of_funds = [
        ("Engineering", "50%", "Managed cloud, enterprise auth, Prometheus metrics, horizontal scaling"),
        ("Go-to-Market", "30%", "Developer relations, enterprise sales, docs, community"),
        ("Operations", "20%", "Infrastructure, security audit, legal, compliance"),
    ]
    for i, (label, pct, desc) in enumerate(use_of_funds):
        x = Inches(0.55 + i * 4.2)
        rect(s, x, Inches(2.9), Inches(3.9), Inches(2.8), DARKCARD)
        rect(s, x, Inches(2.9), Inches(3.9), Pt(4), BLUE)
        tb = txbox(s, x + Inches(0.2), Inches(3.1), Inches(3.5), Inches(0.7))
        tf = tb.text_frame
        para(tf, f"{label}  {pct}", 20, bold=True, color=WHITE)
        tb = txbox(s, x + Inches(0.2), Inches(3.85), Inches(3.5), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, desc, 12, color=GREY)

    # Contact
    rect(s, 0, Inches(6.0), W, Inches(1.5), DARKCARD)
    tb = txbox(s, Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.5))
    tf = tb.text_frame
    para(tf, "s.lukov@dynamicpricing.ai  ·  dynamicpricing.ai  ·  github.com/dynamicpricing-ai/banditdb", 14, color=WHITE, align=PP_ALIGN.CENTER)

    tb = txbox(s, Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.5))
    tf = tb.text_frame
    para(tf, "Try it now:  curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh", 11, color=BLUE, align=PP_ALIGN.CENTER)
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = "Courier New"

# ── Slide 5: Observability ────────────────────────────────────────────────────
def slide_observability(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.3), Inches(11), Inches(0.85))
    tf = tb.text_frame
    para(tf, "You always know what your model is doing — and why.", 28, bold=True, color=WHITE)

    tb = txbox(s, Inches(0.7), Inches(1.1), Inches(11), Inches(0.45))
    tf = tb.text_frame
    para(tf, "Most ML models are black boxes. BanditDB ships full observability out of the box.", 14, color=GREY)

    # Left column — feature list
    features = [
        (BLUE,   "Entropy alerting",
                 "Detects arm collapse before it hurts revenue.\nOK / Warning / Critical — live."),
        (BLUE,   "Convergence signal",
                 "95% CI per arm. Tells you exactly when\nthe experiment is done."),
        (ORANGE, "Full audit trail",
                 "WAL + JSONL audit log. Every predict and\nreward recorded for compliance."),
        (BLUE,   "Parquet export",
                 "Every checkpoint exports matched\npredict→reward pairs. Analyse anywhere."),
    ]
    for i, (color, title, body) in enumerate(features):
        y = Inches(1.75 + i * 1.32)
        rect(s, Inches(0.6), y, Pt(4), Inches(1.15), color)
        tb = txbox(s, Inches(0.95), y + Inches(0.05), Inches(5.2), Inches(0.45))
        tf = tb.text_frame
        para(tf, title, 14, bold=True, color=WHITE)
        tb = txbox(s, Inches(0.95), y + Inches(0.48), Inches(5.2), Inches(0.65))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, body, 12, color=GREY)

    # Right column — mock API response
    rect(s, Inches(6.5), Inches(1.65), Inches(6.55), Inches(5.35), DARKCARD)
    rect(s, Inches(6.5), Inches(1.65), Inches(6.55), Pt(4), ORANGE)

    tb = txbox(s, Inches(6.7), Inches(1.75), Inches(6.1), Inches(0.4))
    tf = tb.text_frame
    para(tf, "GET /campaign/churn/diagnostics", 11, bold=True, color=ORANGE)
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.name = "Courier New"

    api_response = (
        '{\n'
        '  "selection_entropy": 0.742,\n'
        '  "entropy_status": "ok",\n'
        '  "entropy_trend": "stable",\n'
        '  "converged": false,\n'
        '  "arm_stats": {\n'
        '    "no_action":   { "predictions": 4231,\n'
        '                     "avg_reward": 0.89 },\n'
        '    "csm_call":    { "predictions": 1847,\n'
        '                     "avg_reward": 0.61 },\n'
        '    "discount":    { "predictions": 2104,\n'
        '                     "avg_reward": 0.74 }\n'
        '  },\n'
        '  "suggested_action": null\n'
        '}'
    )
    tb = txbox(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = False
    para(tf, api_response, 11, color=WHITE)
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.name = "Courier New"

    # Highlight two values in the mock response
    tb = txbox(s, Inches(7.65), Inches(2.35), Inches(1.6), Inches(0.28))
    tf = tb.text_frame
    para(tf, "0.742", 11, color=BLUE, bold=True)
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.name = "Courier New"

    # Bottom strip
    rect(s, 0, Inches(7.1), W, Inches(0.4), DARKCARD)
    tb = txbox(s, Inches(0.5), Inches(7.13), Inches(12.3), Inches(0.3))
    tf = tb.text_frame
    para(tf, "Enterprise-ready: RBAC · Rate limiting · Multi-tenancy · Audit log · OpenAPI spec",
         11, color=GREY, align=PP_ALIGN.CENTER)


# ── Slide 6: Architecture / A different database ──────────────────────────────
def slide_architecture(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.28), Inches(11.9), Inches(0.7))
    tf = tb.text_frame
    para(tf, "Not a vector database. A fundamentally different memory model.", 26, bold=True, color=WHITE)

    tb = txbox(s, Inches(0.7), Inches(1.05), Inches(11.9), Inches(0.42))
    tf = tb.text_frame
    para(tf, "Vector DBs answer: \"what's similar?\"    BanditDB answers: \"what works?\"    And it never forgets the difference.", 13, color=GREY)

    # ── Comparison table ──────────────────────────────────────────────────────
    headers = ["", "Vector Database", "BanditDB"]
    rows = [
        ("Storage growth",    "O(n) — grows with every document",      "O(d²) — fixed size forever"),
        ("After 1M events",   "Millions of stored vectors",            "One 32×32 matrix per arm  (8 KB)"),
        ("Purpose",           "Find similar items",                    "Make optimal decisions"),
        ("Learning",          "None — static index",                   "Continuous — every reward updates"),
        ("Handles drift",     "Delete & re-index old vectors",         "Decay parameter fades old signal"),
        ("Query type",        "Nearest neighbour search",              "Predict best action for context"),
    ]

    col_w = [Inches(2.8), Inches(4.5), Inches(4.5)]
    col_x = [Inches(0.55), Inches(3.35), Inches(7.85)]
    row_h = Inches(0.62)
    row_y0 = Inches(1.68)

    # Header row
    for ci, (txt, cw, cx) in enumerate(zip(headers, col_w, col_x)):
        bg_col = BLUE if ci == 2 else DARKCARD
        rect(s, cx, row_y0, cw, row_h, bg_col)
        tb = txbox(s, cx + Inches(0.12), row_y0 + Inches(0.1), cw - Inches(0.2), row_h - Inches(0.15))
        tf = tb.text_frame
        c = WHITE if txt else NAVY
        para(tf, txt, 13, bold=True, color=c)

    # Data rows
    for ri, row in enumerate(rows):
        y = row_y0 + row_h + ri * row_h
        for ci, (txt, cw, cx) in enumerate(zip(row, col_w, col_x)):
            bg_col = RGBColor(0x0C, 0x16, 0x28) if ri % 2 == 0 else DARKCARD
            if ci == 2:
                bg_col = RGBColor(0x05, 0x18, 0x2E) if ri % 2 == 0 else RGBColor(0x07, 0x20, 0x3A)
            rect(s, cx, y, cw, row_h, bg_col)
            col = BLUE if ci == 2 else (GREY if ci == 1 else WHITE)
            bold = ci == 0
            tb = txbox(s, cx + Inches(0.12), y + Inches(0.1), cw - Inches(0.2), row_h - Inches(0.15))
            tf = tb.text_frame
            tf.word_wrap = True
            para(tf, txt, 12, bold=bold, color=col)

    # ── Condensed knowledge callout ───────────────────────────────────────────
    rect(s, 0, Inches(6.3), W, Inches(1.2), DARKCARD)
    rect(s, 0, Inches(6.3), W, Pt(4), ORANGE)

    tb = txbox(s, Inches(0.7), Inches(6.4), Inches(5.5), Inches(0.4))
    tf = tb.text_frame
    para(tf, "Condensed knowledge, not raw data storage", 14, bold=True, color=ORANGE)

    tb = txbox(s, Inches(0.7), Inches(6.8), Inches(5.5), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    para(tf, "A_inv and θ compress 1M interactions into a fixed 32×32 matrix.\nMore data makes it smarter — not bigger.", 11, color=GREY)

    # Divider
    rect(s, Inches(6.3), Inches(6.35), Pt(1), Inches(1.0), RGBColor(0x20, 0x35, 0x55))

    tb = txbox(s, Inches(6.55), Inches(6.4), Inches(5.8), Inches(0.4))
    tf = tb.text_frame
    para(tf, "Adaptive forgetting — stays current automatically", 14, bold=True, color=BLUE)

    tb = txbox(s, Inches(6.55), Inches(6.8), Inches(6.5), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    para(tf, "decay_half_life_hours fades old signal at every checkpoint.\nNo manual re-training. No stale models. Adapts to concept drift.", 11, color=GREY)


# ── Slide 9: Business Model ───────────────────────────────────────────────────
def slide_business_model(prs):
    s = blank_slide(prs)
    bg(s)
    accent_line(s, Inches(0.7), Inches(1.2), Inches(0.5))

    tb = txbox(s, Inches(0.7), Inches(0.28), Inches(11.9), Inches(0.7))
    tf = tb.text_frame
    para(tf, "Open-core model. Proven at scale by MongoDB, Elastic, and Redis.", 26, bold=True, color=WHITE)

    tb = txbox(s, Inches(0.7), Inches(1.05), Inches(11.9), Inches(0.42))
    tf = tb.text_frame
    para(tf, "Open source drives adoption → Cloud converts power users → Enterprise converts revenue.", 13, color=GREY)

    # Three pricing cards
    tiers = [
        {
            "label":    "OPEN SOURCE",
            "name":     "Self-Hosted",
            "price":    "Free",
            "price_sub":"forever · Apache 2.0",
            "accent":   RGBColor(0x45, 0x56, 0x6B),
            "tag":      None,
            "features": [
                "All 5 algorithms",
                "WAL + checkpoint durability",
                "Full HTTP API",
                "RBAC + rate limiting",
                "Parquet export",
                "MCP server for AI agents",
                "Community support",
            ],
            "cta":      "github.com/dynamicpricing-ai/banditdb",
        },
        {
            "label":    "BANDITDB CLOUD",
            "name":     "Managed Cloud",
            "price":    "Usage-based",
            "price_sub":"from $0.0001 / prediction",
            "accent":   BLUE,
            "tag":      "MOST POPULAR",
            "features": [
                "Everything in Open Source",
                "Zero infrastructure",
                "Auto-scaling to 100K+ req/sec",
                "Web dashboard + analytics",
                "Automatic backups",
                "99.9% SLA",
                "Email + chat support",
            ],
            "cta":      "banditdb.cloud  (coming soon)",
        },
        {
            "label":    "ENTERPRISE",
            "name":     "Private Deployment",
            "price":    "Custom",
            "price_sub":"annual contract",
            "accent":   ORANGE,
            "tag":      None,
            "features": [
                "Everything in Cloud",
                "On-premise or VPC deploy",
                "SSO / SAML",
                "SOC 2 · HIPAA · GDPR",
                "Dedicated SLA (99.99%)",
                "Custom integrations",
                "Dedicated success engineer",
            ],
            "cta":      "Contact  s.lukov@dynamicpricing.ai",
        },
    ]

    card_w = Inches(4.0)
    card_h = Inches(5.15)
    gap    = Inches(0.27)
    x0     = Inches(0.55)

    for i, tier in enumerate(tiers):
        x     = x0 + i * (card_w + gap)
        y_top = Inches(1.65)
        accent = tier["accent"]
        is_mid = i == 1

        # Card background — middle card slightly brighter
        card_bg = RGBColor(0x0E, 0x1E, 0x38) if is_mid else DARKCARD
        rect(s, x, y_top, card_w, card_h, card_bg)
        rect(s, x, y_top, card_w, Pt(5), accent)

        # "Most popular" badge
        if tier["tag"]:
            rect(s, x + Inches(1.0), y_top - Inches(0.28), Inches(2.0), Inches(0.28), accent)
            tb = txbox(s, x + Inches(1.0), y_top - Inches(0.28), Inches(2.0), Inches(0.26))
            tf = tb.text_frame
            para(tf, tier["tag"], 9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        # Tier label
        tb = txbox(s, x + Inches(0.18), y_top + Inches(0.12), card_w - Inches(0.3), Inches(0.3))
        tf = tb.text_frame
        para(tf, tier["label"], 9, bold=True, color=accent)

        # Name
        tb = txbox(s, x + Inches(0.18), y_top + Inches(0.42), card_w - Inches(0.3), Inches(0.55))
        tf = tb.text_frame
        para(tf, tier["name"], 20, bold=True, color=WHITE)

        # Price
        tb = txbox(s, x + Inches(0.18), y_top + Inches(0.96), card_w - Inches(0.3), Inches(0.55))
        tf = tb.text_frame
        para(tf, tier["price"], 26, bold=True, color=accent)

        tb = txbox(s, x + Inches(0.18), y_top + Inches(1.48), card_w - Inches(0.3), Inches(0.28))
        tf = tb.text_frame
        para(tf, tier["price_sub"], 10, color=GREY)

        # Divider
        rect(s, x + Inches(0.18), y_top + Inches(1.82), card_w - Inches(0.36), Pt(1),
             RGBColor(0x20, 0x35, 0x55))

        # Features
        for j, feat in enumerate(tier["features"]):
            fy = y_top + Inches(1.98) + j * Inches(0.38)
            tb = txbox(s, x + Inches(0.18), fy, card_w - Inches(0.3), Inches(0.36))
            tf = tb.text_frame
            para(tf, f"✓  {feat}", 11, color=WHITE if is_mid else GREY)

        # CTA
        tb = txbox(s, x + Inches(0.18), y_top + card_h - Inches(0.38), card_w - Inches(0.3), Inches(0.32))
        tf = tb.text_frame
        para(tf, tier["cta"], 9, color=accent)
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.name = "Courier New"

    # Revenue flywheel note at bottom
    rect(s, 0, Inches(7.0), W, Inches(0.5), RGBColor(0x05, 0x10, 0x20))
    tb = txbox(s, Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.38))
    tf = tb.text_frame
    para(tf,
         "Open source → developer trust → viral adoption → cloud conversion → enterprise contracts  "
         "·  Same flywheel that built MongoDB ($24B), Elastic ($8B), and Confluent ($9B)",
         10, color=GREY, align=PP_ALIGN.CENTER)


# ── Build ──────────────────────────────────────────────────────────────────────
def main():
    prs = new_prs()
    slide_hero(prs)
    slide_problem(prs)
    slide_solution(prs)
    slide_how(prs)
    slide_observability(prs)
    slide_architecture(prs)
    slide_traction(prs)
    slide_market(prs)
    slide_why_now(prs)
    slide_business_model(prs)
    slide_ask(prs)
    prs.save(OUT_PATH)
    print(f"Saved → {OUT_PATH}  ({len(prs.slides)} slides)")

if __name__ == "__main__":
    main()
