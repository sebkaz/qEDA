"""Build the plain-language qEDA computation summary PDF.

The report reads the frozen CSV artifacts shipped with the reproducibility
package.  Mathematical notation is written in an ASCII form so the document
remains searchable and robust across PDF viewers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "pdf" / "qeda_computation_summary.pdf"
DATA = ROOT / "results" / "data"
BENCHMARK = ROOT / "results" / "benchmark"
FIGURES = ROOT / "results" / "figures"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#147D73")
LIGHT_TEAL = colors.HexColor("#E7F4F1")
ORANGE = colors.HexColor("#D97732")
LIGHT_ORANGE = colors.HexColor("#FFF1E7")
INK = colors.HexColor("#24303B")
MUTED = colors.HexColor("#64717D")
GRID = colors.HexColor("#D5DDE4")
LIGHT = colors.HexColor("#F5F7F9")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str]:
    """Register stable system fonts and return their ReportLab names."""
    font_dir = Path("/System/Library/Fonts/Supplemental")
    candidates = {
        "QedaSans": font_dir / "Verdana.ttf",
        "QedaSansBold": font_dir / "Verdana Bold.ttf",
        "QedaSansItalic": font_dir / "Verdana Italic.ttf",
        "QedaMono": font_dir / "Courier New.ttf",
    }
    if all(path.exists() for path in candidates.values()):
        for name, path in candidates.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        return "QedaSans", "QedaSansBold", "QedaSansItalic", "QedaMono"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


FONT, FONT_BOLD, FONT_ITALIC, FONT_MONO = register_fonts()


class NumberedCanvas(Canvas):
    """Canvas that adds a restrained header and page x of y footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(page_count)
            super().showPage()
        super().save()

    def _draw_header_footer(self, page_count: int) -> None:
        page_number = self._pageNumber
        self.saveState()
        if page_number > 1:
            self.setStrokeColor(GRID)
            self.setLineWidth(0.5)
            self.line(MARGIN, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 13 * mm)
            self.setFont(FONT_BOLD, 7.5)
            self.setFillColor(NAVY)
            self.drawString(MARGIN, PAGE_HEIGHT - 10 * mm, "qEDA computation guide")
        self.setStrokeColor(GRID)
        self.line(MARGIN, 12 * mm, PAGE_WIDTH - MARGIN, 12 * mm)
        self.setFont(FONT, 7)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, 8 * mm, "Reproducibility companion - 11 August 2026")
        self.drawRightString(
            PAGE_WIDTH - MARGIN,
            8 * mm,
            f"Page {page_number} of {page_count}",
        )
        self.restoreState()


class Rule(Flowable):
    """Horizontal rule with controlled spacing."""

    def __init__(self, color: colors.Color = GRID, thickness: float = 0.6):
        super().__init__()
        self.color = color
        self.thickness = thickness
        self.width = CONTENT_WIDTH
        self.height = 5 * mm

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.height / 2, self.width, self.height / 2)


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CoverKicker",
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        textColor=TEAL,
        alignment=TA_CENTER,
        spaceAfter=7 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverTitle",
        fontName=FONT_BOLD,
        fontSize=25,
        leading=31,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverSub",
        fontName=FONT,
        fontSize=11,
        leading=17,
        textColor=INK,
        alignment=TA_CENTER,
        spaceAfter=9 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="H1q",
        fontName=FONT_BOLD,
        fontSize=16,
        leading=20,
        textColor=NAVY,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="H2q",
        fontName=FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=TEAL,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Bodyq",
        fontName=FONT,
        fontSize=8.7,
        leading=13.2,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=2.3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Smallq",
        fontName=FONT,
        fontSize=7.5,
        leading=10.5,
        textColor=INK,
        spaceAfter=1.6 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Captionq",
        fontName=FONT,
        fontSize=7.2,
        leading=10,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=1.5 * mm,
        spaceAfter=3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Codeq",
        fontName=FONT_MONO,
        fontSize=7.8,
        leading=11.5,
        textColor=NAVY,
        leftIndent=2 * mm,
        rightIndent=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHeadq",
        fontName=FONT_BOLD,
        fontSize=6.8,
        leading=8.5,
        textColor=WHITE,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCellq",
        fontName=FONT,
        fontSize=6.7,
        leading=8.8,
        textColor=INK,
    )
)


def p(text: str, style: str = "Bodyq") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(text, styles["Bodyq"], bulletText="-")


def callout(title: str, body: str, tone: str = "teal") -> Table:
    background = LIGHT_TEAL if tone == "teal" else LIGHT_ORANGE
    accent = TEAL if tone == "teal" else ORANGE
    content = Paragraph(
        f'<font name="{FONT_BOLD}" color="{accent.hexval()}">{title}</font><br/>{body}',
        ParagraphStyle(
            "CalloutInner",
            parent=styles["Bodyq"],
            spaceAfter=0,
            leftIndent=0,
        ),
    )
    table = Table([[content]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def code_box(lines: list[str]) -> Table:
    paragraph = Paragraph("<br/>".join(lines), styles["Codeq"])
    table = Table([[paragraph]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def data_table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    font_size: float = 6.7,
) -> Table:
    header_cells = [p(value, "TableHeadq") for value in headers]
    cell_style = ParagraphStyle(
        "DynamicCell",
        parent=styles["TableCellq"],
        fontSize=font_size,
        leading=font_size + 2.0,
    )
    body = [[Paragraph(str(value), cell_style) for value in row] for row in rows]
    table = Table([header_cells] + body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_number in range(1, len(rows) + 1):
        if row_number % 2 == 0:
            commands.append(("BACKGROUND", (0, row_number), (-1, row_number), LIGHT))
    table.setStyle(TableStyle(commands))
    return table


def fitted_image(path: Path, max_width: float, max_height: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def read_results() -> dict[str, pd.DataFrame]:
    """Load every frozen table used in the report."""
    return {
        "algebra": pd.read_csv(DATA / "algebraic_controls.csv"),
        "topology_profiles": pd.read_csv(DATA / "topology_holonomy_profiles.csv"),
        "topology_permutation": pd.read_csv(DATA / "topology_holonomy_permutation.csv"),
        "topology_loops": pd.read_csv(DATA / "topology_holonomy_loops.csv"),
        "benchmark": pd.read_csv(BENCHMARK / "eda_comparison.csv"),
        "coverage": pd.read_csv(BENCHMARK / "spectral_coverage_pennylane.csv"),
        "iris": pd.read_csv(DATA / "iris_audit_summary.csv"),
        "fraud_paths": pd.read_csv(DATA / "fraud_qeda_paths.csv"),
        "fraud": pd.read_csv(DATA / "fraud_qeda_summary.csv"),
    }


def build_story(results: dict[str, pd.DataFrame]) -> list[Flowable]:
    story: list[Flowable] = []

    # Cover
    story.extend(
        [
            Spacer(1, 22 * mm),
            p("NUMERICAL COMPANION", "CoverKicker"),
            p("qEDA computations:<br/>what is calculated, where, and what it shows", "CoverTitle"),
            p(
                "A plain-language guide to the algebraic controls, matched topology test, "
                "Bowles benchmark families, Iris audit, and credit-card fraud case study.",
                "CoverSub",
            ),
            Spacer(1, 5 * mm),
            callout(
                "The central question",
                "After a table is embedded into a Hilbert space, does the resulting "
                "operator reproduce a declared classical view, add structure irrelevant "
                "to the analyst's question, or supply a stable additional description?",
            ),
            Spacer(1, 7 * mm),
            data_table(
                ["Audit", "Data", "Main answer"],
                [
                    ["E1 controls", "Synthetic states", "The implementation satisfies the exact operator and phase identities."],
                    ["E2 topology", "Ring vs filled disk", "Quantum-looking structure appears, but it does not detect the support hole."],
                    ["E3 benchmark", "55 Bowles variants", "Spectral response is not a simple relabelling of the selected covariance summaries."],
                    ["E4 Iris", "150 native rows", "Rows missed by classical probes have lower held-out class-subspace fidelity."],
                    ["E4 fraud", "284,807 released rows", "qEDA ranks anomalies well; partial J adds almost no ranking information beyond J = 0."],
                ],
                [31 * mm, 45 * mm, 98 * mm],
                font_size=7.1,
            ),
            Spacer(1, 7 * mm),
            p(
                "Companion to <i>Beyond the kernel: exploratory analysis of quantum "
                "encodings of tabular data through the class density operator</i>.<br/>"
                "Sebastian Zajac and Jacob L. Cybulski - 11 August 2026",
                "Captionq",
            ),
            PageBreak(),
        ]
    )

    # Common pipeline
    story.extend(
        [
            p("1. The common computation", "H1q"),
            p(
                "Each row x is first mapped to a normalized state |psi(x,J)>. The rows "
                "are not compressed by PCA to satisfy a qubit budget. Standardisation is "
                "used to estimate dependence, while a separate train-fitted affine map "
                "places native features in the angle interval [0, pi].",
            ),
            code_box(
                [
                    "input table -> fitted scaling -> angles x",
                    "x -> RY(x/2) -> ZZ(J) -> RY(x/2) -> |psi(x,J)>",
                    "class rows -> rho_c = (1/M_c) sum_m |psi_m><psi_m|",
                    "rho_c -> spectrum, subsystems, mode coherence, row scores",
                    "reported result = comparison with matched controls",
                ]
            ),
            p("Matched encodings", "H2q"),
            p(
                "The product control and the coupled map have the same rows, angle map, "
                "and two local half-layers. Only the interaction changes. At J = 0 the "
                "two RY(x/2) layers compose to the ordinary real product RY(x) encoding.",
            ),
            data_table(
                ["Encoding", "Definition", "Purpose"],
                [
                    ["J = 0", "RY(x/2) - I - RY(x/2)", "Real product-angle baseline."],
                    ["Partial J", "RY(x/2) - ZZ(J) - RY(x/2)", "Adds deterministic conditional-dependence interactions."],
                ],
                [28 * mm, 67 * mm, 79 * mm],
            ),
            p("How J is obtained", "H2q"),
            code_box(
                [
                    "Sigma = covariance of standardised analysis rows",
                    "Theta = inverse(Sigma + epsilon I)",
                    "J_jk = -s Theta_jk / sqrt(Theta_jj Theta_kk),  j != k",
                    "J_jj = 0",
                ]
            ),
            bullet("J is not learned from accuracy or a predictive loss."),
            bullet("There is no dataset-wise rescaling that forces max |J_jk| = 1."),
            bullet("The scale is declared before an audit: s = 1.0 for Bowles and s = 0.8 for topology, Iris, and fraud."),
            bullet("Every held-out audit fits scaling, J, rho, and thresholds without using held-out labels."),
            Spacer(1, 2 * mm),
            callout(
                "Simple reading",
                "We deliberately give the Hilbert-space representation a fair chance to "
                "add structure, then ask whether that structure is stable and relevant. "
                "A nonzero quantum diagnostic is not automatically a useful data result.",
                tone="orange",
            ),
            PageBreak(),
        ]
    )

    # Diagnostics
    story.extend(
        [
            p("2. What the reported numbers mean", "H1q"),
            p(
                "Diagonalising rho gives nonnegative weights lambda_0 >= lambda_1 >= ... "
                "that sum to one. These weights describe how concentrated the encoded "
                "empirical mixture is. Subsystem and mode-coherence readings ask different "
                "questions and must not be collapsed into a single score.",
            ),
            data_table(
                ["Reading", "Calculation", "Plain interpretation"],
                [
                    ["Purity", "sum_i lambda_i^2", "Larger means the encoded class mixture is concentrated into fewer directions."],
                    ["Effective rank", "1 / purity", "Approximate number of substantially occupied spectral directions."],
                    ["Mass gap", "log(lambda_0 / lambda_1)", "Separation between the leading direction and its nearest competitor."],
                    ["Third moment", "Tr(rho^3)", "Another concentration summary, more sensitive to dominant eigenvalues."],
                    ["Mutual information I", "Mean single-mode/rest entropy balance", "Total dependence across one encoded mode versus the remaining modes."],
                    ["Log negativity Q", "Mean partial-transpose trace norm", "Entanglement witness across single-mode cuts; Q > 0 certifies representation entanglement."],
                    ["Encoded current", "max |Im Gamma_jk|", "Basis-explicit antisymmetric mode coherence. It is not an entanglement witness."],
                    ["Row fidelity Fs", "<psi(x)|P_c|psi(x)>", "How much of a row lies in the leading subspace learned for class c."],
                    ["Anomaly score", "1 - Fs", "Large values mean a row lies outside the leading normal-reference subspace."],
                    ["Bargmann phase", "arg product of closed-loop overlaps", "Gauge-invariant geometric phase of a chosen loop; not automatically topological."],
                ],
                [32 * mm, 52 * mm, 90 * mm],
                font_size=6.5,
            ),
            p("Why the ordinary quantum kernel is not enough", "H2q"),
            p(
                "Let A contain the statevectors as columns and G = A^dagger A be the "
                "complex Gram matrix. The nonzero spectrum of rho = A A^dagger / M equals "
                "the spectrum of G/M. Pairwise overlaps therefore retain the operator "
                "spectrum. Information is discarded when the standard fidelity kernel "
                "replaces G_mn by |G_mn|^2, because overlap phases and closed-loop phase "
                "products disappear.",
            ),
            callout(
                "Interpretation rule",
                "Purity, Q, or current can show that an encoding created operator "
                "structure. Relevance requires another control: stability, a null test, "
                "or a held-out data question. The report always states both layers.",
            ),
            PageBreak(),
        ]
    )

    # Data map
    story.extend(
        [
            p("3. Where each computation is used", "H1q"),
            data_table(
                ["Data", "Size / features", "Fit and control", "Question"],
                [
                    ["Synthetic exact checks", "Small 3-qubit tables", "Same rows under J = 0 and coupled maps", "Does the code obey the algebraic identities and phase conventions?"],
                    ["Ring vs disk", "300 + 300 rows; 3 features", "Means and covariance matched to machine precision; 500 permutations", "Do current summaries reveal a known support hole beyond first two moments?"],
                    ["Bowles families", "55 variants; 2-16 native features", "Native classical EDA plus matched J = 0 / J; no PCA", "Does qEDA provide an ordering distinct from selected covariance summaries?"],
                    ["Iris", "150 rows; 4 native features", "Five-fold cross-fitting; three classical error probes", "Are hard held-out rows also less compatible with the true-class operator?"],
                    ["Credit-card fraud", "284,807 rows; 30 released variables", "Normal-only fit and calibration; natural-prevalence test", "Can a normal-reference operator score anomalies, and does coupling add information?"],
                ],
                [31 * mm, 31 * mm, 56 * mm, 56 * mm],
                font_size=6.4,
            ),
            p("High-dimensional handling in the fraud audit", "H2q"),
            p(
                "Thirty variables do not fit the declared four-qubit full-operator budget. "
                "No XGBoost ranking, predictive importance, PCA, or first-column rule is "
                "used. Four seeded permutations and all 30 cyclic shifts create 120 paths. "
                "Each path uses four variables, and every variable appears exactly 16 "
                "times. Names are sorted before wire assignment, so stored column order "
                "does not choose the result.",
            ),
            code_box(
                [
                    "30 variables x 4 cyclic base paths = 120 four-variable marginals",
                    "each variable occurs 4 x 4 = 16 times",
                    "path score -> aggregate by median (primary) or 90th percentile (sensitivity)",
                ]
            ),
            p("Fraud split", "H2q"),
            bullet("A stratified 80/20 split gives 56,962 held-out transactions, including 98 frauds."),
            bullet("5,000 normal training rows fit scaling, J, rho, and the leading projector."),
            bullet("A disjoint 10,000 normal rows calibrate target false-positive rates."),
            bullet("Fraud labels are used only for held-out evaluation."),
            p("Important data limitation", "H2q"),
            p(
                "The released variables V1-V28 were already anonymised by PCA before "
                "publication. We apply no additional PCA. The conclusions concern the "
                "released representation, not the unavailable original transaction fields.",
            ),
            PageBreak(),
        ]
    )

    # E1
    algebra = results["algebra"]
    algebra_rows = []
    for row in algebra.itertuples(index=False):
        algebra_rows.append(
            [
                str(row.experiment).replace("_", " "),
                str(row.metric).replace("_", " "),
                f"{float(row.J0):.4g}",
                f"{float(row.J_partial):.4g}",
                str(row.notes),
            ]
        )
    story.extend(
        [
            p("4. E1 - algebraic and implementation controls", "H1q"),
            p(
                "These checks are not data discoveries. They verify that later differences "
                "are produced by the declared map rather than a sign, normalisation, or "
                "PennyLane convention error. The vectorised fraud encoder is additionally "
                "checked state by state against PennyLane; its maximum amplitude error is "
                "2.22e-16.",
            ),
            data_table(
                ["Experiment", "Metric", "J = 0", "Partial J", "Meaning"],
                algebra_rows,
                [27 * mm, 43 * mm, 20 * mm, 22 * mm, 62 * mm],
                font_size=6.2,
            ),
            p("What this establishes", "H2q"),
            bullet("The class-operator spectrum and scaled complex Gram spectrum agree to machine precision."),
            bullet("The real J = 0 control has no imaginary overlap structure and zero encoded current in the declared basis."),
            bullet("The coupled map can create overlap phase, entanglement, mutual information, and current."),
            bullet("The last statement concerns the representation. It does not yet say that the added structure answers a data question."),
            callout(
                "E1 conclusion",
                "The implementation passes the required identities. We can therefore "
                "interpret later negative results as genuine audit outcomes, not as a "
                "failure to generate quantum operator structure.",
            ),
            PageBreak(),
        ]
    )

    # E2 topology
    topo_p = results["topology_permutation"]
    p_pivot = topo_p.pivot(index="metric", columns="encoding", values="permutation_p_two_sided")
    metric_order = ["purity", "mass_gap", "third_moment", "I", "Q", "current"]
    topo_rows = [
        [
            metric.replace("_", " "),
            f"{p_pivot.loc[metric, 'J0']:.3f}",
            f"{p_pivot.loc[metric, 'J_partial']:.3f}",
        ]
        for metric in metric_order
    ]
    profiles = results["topology_profiles"]
    coupled = profiles[profiles["encoding"] == "J_partial"]
    q_mean = coupled["Q"].mean()
    current_mean = coupled["current"].mean()
    loops = results["topology_loops"]
    coupled_loops = loops[loops["encoding"] == "J_partial"].sort_values("radius")
    story.extend(
        [
            p("5. E2 - matched-moment topology audit", "H1q"),
            p(
                "The ring and filled disk have visibly different supports, but separate "
                "invertible recolouring makes their empirical means and covariance matrices "
                "equal to 1.49e-16 and 2.00e-15. The test asks whether the current operator "
                "summaries recover the hole after this classical second-order information "
                "has been matched.",
            ),
            fitted_image(FIGURES / "topology_holonomy_test.png", CONTENT_WIDTH, 107 * mm),
            p(
                "Figure 1. The clouds have matched first two moments. Coupling creates a "
                "geometric loop phase and nonzero operator structure, but the permutation "
                "statistics remain inside their null bands.",
                "Captionq",
            ),
            data_table(
                ["Metric", "p, J = 0", "p, partial J"],
                topo_rows,
                [64 * mm, 50 * mm, 60 * mm],
                font_size=7.1,
            ),
            Spacer(1, 2 * mm),
            p(
                f"The coupled operators are genuinely enriched: mean Q is {q_mean:.3f} "
                f"and mean maximum current is {current_mean:.3f}. Nevertheless, every "
                "two-sided permutation p-value exceeds 0.22. The coupled Bargmann phase "
                f"changes continuously from {coupled_loops.iloc[0].bargmann_phase:.2e} to "
                f"{coupled_loops.iloc[-1].bargmann_phase:.2e} radians as the loop expands. "
                "It tends to zero when the loop contracts.",
            ),
            callout(
                "E2 conclusion: a useful negative result",
                "The encoding creates entanglement, current, and geometric holonomy, but "
                "none of the reported statistics detects the support hole. Geometric phase "
                "is not a protected topological invariant here. More quantum structure does "
                "not imply more relevant data information.",
                tone="orange",
            ),
            PageBreak(),
        ]
    )

    # E3 benchmark
    benchmark = results["benchmark"]
    columns = [
        "mean_abs_correlation",
        "correlation_rank_fraction",
        "fisher_trace_ratio",
        "delta_purity",
        "delta_mass_gap",
    ]
    medians = benchmark.groupby("family")[columns].median(numeric_only=True)
    labels = {
        "bars_and_stripes": "Bars and Stripes",
        "hidden_manifold": "Hidden manifold",
        "hidden_manifold_diff": "Hidden manifold (difficulty)",
        "hyperplanes_diff": "Hyperplanes parity",
        "linearly_separable": "Linearly separable",
        "two_curves": "Two curves",
        "two_curves_diff": "Two curves (difficulty)",
    }
    benchmark_rows = []
    for family in labels:
        row = medians.loc[family]
        dp = "-" if pd.isna(row.delta_purity) else f"{row.delta_purity:.3f}"
        dg = "-" if pd.isna(row.delta_mass_gap) else f"{row.delta_mass_gap:.3f}"
        benchmark_rows.append(
            [
                labels[family],
                f"{row.mean_abs_correlation:.3f}",
                f"{row.correlation_rank_fraction:.3f}",
                f"{row.fisher_trace_ratio:.3f}",
                dp,
                dg,
            ]
        )
    profiled = int((results["coverage"]["status"] == "profiled").sum())
    story.extend(
        [
            p("6. E3 - Bowles benchmark families", "H1q"),
            p(
                "The original benchmark generators and random streams are replayed, but "
                "the generated tables - not the published models - are the objects of "
                "study. Standard EDA is calculated on all native features. The correlation "
                "spectrum is inspected but no principal-component scores are formed and no "
                "directions are removed.",
            ),
            p(
                f"All 55 variants receive standard EDA. Full class operators are computed "
                f"for {profiled} variants, producing 108 binary-class operators on 2-10 "
                "qubits. The 4 x 4 Bars-and-Stripes table retains all 16 features and is "
                "explicitly marked outside the full-operator budget.",
            ),
            data_table(
                ["Family median", "Mean |r|", "Corr. rank/d", "Fisher", "Delta purity", "Delta gap"],
                benchmark_rows,
                [51 * mm, 23 * mm, 27 * mm, 22 * mm, 25 * mm, 26 * mm],
                font_size=6.2,
            ),
            p("How to read the table", "H2q"),
            bullet("Mean |r| and correlation rank/d are native-feature summaries."),
            bullet("Delta means partial-J minus the matched J = 0 operator on the same rows."),
            bullet("Positive Delta purity and Delta gap mean the coupled operator concentrates weight and separates its leading mode."),
            p(
                "The linearly separable family has the weakest median marginal correlation "
                "(0.051) and an almost full native correlation rank (0.973), yet it has the "
                "largest median spectral sharpening (Delta purity 0.289; Delta gap 1.093). "
                "Across the 54 profiled variants, Spearman correlation between mean |r| and "
                "Delta purity is -0.234, and with Delta gap it is -0.420.",
            ),
            callout(
                "E3 conclusion",
                "For these declared summaries, qEDA is not merely renaming mean correlation "
                "or Fisher separation. This is relative empirical nonredundancy, not a claim "
                "that no classical statistic could reproduce the same ordering.",
            ),
            PageBreak(),
        ]
    )

    # E4 Iris and fraud results
    iris = results["iris"].iloc[0]
    fraud = results["fraud"]
    selected_fraud = fraud[fraud["target_fpr"].round(6) == 0.005].copy()
    fraud_order = ["qeda_j0_median", "qeda_j_median", "isolation_forest"]
    fraud_labels = {
        "qeda_j0_median": "qEDA J = 0 median",
        "qeda_j_median": "qEDA partial J median",
        "isolation_forest": "Isolation Forest",
    }
    fraud_rows = []
    for method in fraud_order:
        row = selected_fraud[selected_fraud["method"] == method].iloc[0]
        fraud_rows.append(
            [
                fraud_labels[method],
                f"{row.roc_auc:.3f}",
                f"{row.average_precision:.3f}",
                f"{row.observed_test_fpr:.4f}",
                f"{row.recall:.3f}",
                f"{row.precision:.3f}",
            ]
        )
    paths = results["fraud_paths"]
    delta_purity = (paths["purity_J_partial"] - paths["purity_J0"]).median()
    delta_auc = (paths["roc_auc_J_partial"] - paths["roc_auc_J0"]).median()
    delta_ap = (
        paths["average_precision_J_partial"] - paths["average_precision_J0"]
    ).median()
    story.extend(
        [
            p("7. E4 - held-out real-data audits", "H1q"),
            p("Iris: sample-level compatibility", "H2q"),
            p(
                "All four native Iris features fit the full operator. In each of five "
                "stratified folds, every transformation and class operator is fitted on "
                "four folds. The held-out row is scored against the leading subspace of its "
                "true-class operator. A separate union of logistic-regression, LDA, and RBF "
                "SVM errors marks difficult rows.",
            ),
            data_table(
                ["Rows", "Probe errors", "Q", "Current", "Fs correct", "Fs wrong", "p"],
                [[
                    f"{int(iris.n_rows)}",
                    f"{int(iris.union_model_errors)}",
                    f"{iris.Q_partial:.3f}",
                    f"{iris.current_partial:.3f}",
                    f"{iris.fidelity_correct:.3f}",
                    f"{iris.fidelity_wrong:.3f}",
                    f"{iris.mann_whitney_p_one_sided:.2e}",
                ]],
                [21 * mm, 27 * mm, 20 * mm, 22 * mm, 27 * mm, 27 * mm, 30 * mm],
                font_size=7.0,
            ),
            p(
                "The seven rows missed by at least one classical probe have lower true-class "
                "fidelity: 0.823 versus 0.933. This supports a sample-level reading on one "
                "small dataset; it is not a general theorem about classifier errors.",
            ),
            p("Credit-card fraud: normal-reference anomaly scoring", "H2q"),
            data_table(
                ["Method", "AUC", "AP", "Observed FPR", "Recall", "Precision"],
                fraud_rows,
                [51 * mm, 20 * mm, 20 * mm, 29 * mm, 26 * mm, 28 * mm],
                font_size=6.8,
            ),
            p(
                "The table uses the median of 120 marginal anomaly scores and a target FPR "
                "of 0.005 calibrated on separate normal training rows. The test set retains "
                "the natural fraud prevalence. Isolation Forest is context only and does not "
                "define qEDA.",
            ),
            fitted_image(FIGURES / "fraud_qeda_case_study.png", CONTENT_WIDTH, 102 * mm),
            p(
                "Figure 2. Both qEDA profiles rank many frauds above normal transactions. "
                "The J = 0 and partial-J curves nearly coincide despite nonzero coupled "
                "negativity and current.",
                "Captionq",
            ),
            PageBreak(),
        ]
    )

    # Final interpretation/reproduction
    story.extend(
        [
            p("8. What the fraud result actually says", "H1q"),
            p(
                "The partial-J representation is not trivial. Across the 120 paths, median "
                f"Q is {paths['Q_J_partial'].median():.4f} and median current is "
                f"{paths['current_J_partial'].median():.4f}; both are zero for the real "
                "control. Yet the added structure does not materially change anomaly "
                "ranking.",
            ),
            data_table(
                ["Path-level comparison", "Median value", "Reading"],
                [
                    ["Delta purity", f"{delta_purity:.3e}", "Almost no spectral concentration change."],
                    ["Delta AUC", f"{delta_auc:.3e}", "No systematic ranking improvement."],
                    ["Delta average precision", f"{delta_ap:.3e}", "No systematic precision-recall improvement."],
                    ["Coupling Frobenius norm", f"{paths['coupling_frobenius'].median():.4f}", "The coupling is present but modest."],
                    ["Maximum |J_jk|", f"{paths['coupling_max_abs'].median():.4f}", "Weak dependence is not inflated to one."],
                ],
                [55 * mm, 33 * mm, 86 * mm],
                font_size=6.8,
            ),
            callout(
                "Audit decision",
                "Keep the Hilbert-space anomaly score if it is useful, but the current "
                "released fraud representation provides no empirical reason to pay for the "
                "coupled layer. The simpler J = 0 encoding is the defensible choice for this "
                "task unless a different, predeclared data question shows a stable gain.",
                tone="orange",
            ),
            p("Reproduction map", "H2q"),
            data_table(
                ["Command", "Primary outputs"],
                [
                    ["python step1_statistics.py", "Console algebraic checks."],
                    ["python scripts/02_topology/topology_holonomy_test.py", "results/data/topology_*.csv and topology figure."],
                    ["python scripts/03_benchmark/spectral_benchmark.py ...", "results/benchmark classical, spectral, and coverage tables."],
                    ["python e4_model_errors.py", "Cross-fitted Iris console summary."],
                    ["python fraud_qeda_case_study.py", "Fraud path, score, summary tables, and figure."],
                ],
                [78 * mm, 96 * mm],
                font_size=6.5,
            ),
            p("Scope boundary", "H2q"),
            bullet("All computations are classical simulations of declared Hilbert-space embeddings."),
            bullet("Results are conditional on preprocessing, encoding, scale, operator readings, and controls."),
            bullet("Nonzero negativity or current certifies representation structure, not usefulness or topology."),
            bullet("The current paper is an operational guide for analysts and QML researchers. QIFT carries the deeper field-theoretic framework."),
            p("Bottom line", "H2q"),
            p(
                "qEDA is valuable even when it recommends against a more elaborate encoding. "
                "Its role is to make the cost and content of a Hilbert-space representation "
                "visible before that representation is hidden inside a model.",
            ),
            Rule(TEAL, 1.0),
            p(
                "Data provenance: synthetic benchmark generators follow Bowles, Ahmed, and "
                "Schuld, <i>Better than classical? The subtle art of benchmarking quantum "
                "machine learning models</i> (arXiv:2403.07059). Iris is supplied by "
                "scikit-learn. The fraud table is the public anonymised ULB/Kaggle release.",
                "Smallq",
            ),
        ]
    )
    return story


def build_pdf() -> Path:
    """Create the final computation guide and return its path."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title="qEDA computations: what is calculated, where, and what it shows",
        author="Sebastian Zajac and Jacob L. Cybulski",
        subject="Plain-language numerical companion to the qEDA manuscript",
    )
    document.build(build_story(read_results()), canvasmaker=NumberedCanvas)
    return OUTPUT


if __name__ == "__main__":
    path = build_pdf()
    print(path)
