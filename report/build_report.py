"""Build the submission report and its original scientific figures.

Run report_analysis.py first. No fitting or TEST reference reads occur here.
All result tables are populated from saved experiment/diagnostic records.
"""
from __future__ import annotations

import argparse
from datetime import date
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / 'code'
FIGURES = HERE / 'figures'
WIDTH, HEIGHT = A4
MARGIN = 48
CONTENT_WIDTH = WIDTH - 2 * MARGIN
CONTENT_HEIGHT = HEIGHT - 100


def read(path):
    return json.loads(Path(path).read_text())


def pipeline_figure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ink, edge = '#17232f', '#73818b'

    def box(x, y, w, h, text, fill='#f3f5f7', size=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.009',
                                   facecolor=fill, edgecolor=edge, linewidth=.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size,
                color=ink, linespacing=1.4)

    def arrow(points):
        if len(points) > 2:
            ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]], color=edge, lw=.95)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                     mutation_scale=9, linewidth=.95, color=edge))

    ax.text(.01, .975, 'FIT ON TRAIN AND FREEZE AFTER VALIDATION SELECTION',
            fontsize=9, fontweight='bold', color=ink, va='top')
    box(.015, .80, .17, .115, 'TRAIN 4,755\nquestion folds')
    box(.235, .80, .23, .115, 'OOF scores + PCA64\nTF-IDF and normalizers')
    box(.515, .80, .20, .115, 'Fit ranking tree\nand attention families')
    box(.765, .80, .22, .115, 'VALIDATION 798\nselect fusion and freeze')
    for left, right in ((.185, .235), (.465, .515), (.715, .765)):
        arrow([(left + .008, .857), (right - .008, .857)])
    ax.plot([.01, .99], [.742, .742], color='#d9dfe3', lw=.8)
    ax.text(.01, .705, 'FROZEN CPU INFERENCE FOR EACH QUESTION', fontsize=9,
            fontweight='bold', color=ink, va='top')
    box(.015, .285, .17, .20, 'Question and\nfour candidate traces\n1024-D embeddings', size=8.0)
    box(.235, .285, .18, .20, 'Arithmetic and text\nfeatures plus\nraw embeddings', size=8.1)
    box(.475, .505, .285, .115, '5 margin scores → tree T\n788-D input | 309 trees', '#eaf1f8', 8.1)
    box(.475, .325, .285, .115, 'Balanced compact attention C\n5 members | 1046-D input', '#eaf3ef', 8.0)
    box(.475, .145, .285, .115, 'Rich attention R\n3 members | 2294-D input', '#f6efe7', 8.1)
    box(.815, .285, .17, .20, 'p = 1/2 T\n+ 1/3 C\n+ 1/6 R', '#eff1f4', 9.0)
    box(.815, .075, .17, .115, 'Same-answer mean\nselect candidate index', size=7.9)
    arrow([(.193, .385), (.227, .385)])
    arrow([(.423, .385), (.446, .385), (.446, .563), (.467, .563)])
    arrow([(.423, .385), (.467, .385)])
    arrow([(.423, .385), (.446, .385), (.446, .203), (.467, .203)])
    arrow([(.768, .563), (.90, .563), (.90, .493)])
    arrow([(.768, .385), (.807, .385)])
    arrow([(.768, .203), (.785, .203), (.785, .325), (.807, .325)])
    arrow([(.90, .277), (.90, .198)])
    ax.text(.015, .08, 'OOF = held-question predictions from TRAIN folds.\nAll three deployed branches and all members are timed.',
            fontsize=7.7, color='#44515b', va='center', linespacing=1.5)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(FIGURES / f'pipeline.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def ablation_figure(metrics):
    rows = [row for row in metrics['ablations'] if row['kind'] != 'training_variant_replacement']
    labels = ['Remove tree', 'Remove balanced compact', 'Remove rich attention',
              'Remove answer pooling']
    values = [-row['delta_correct_vs_full'] for row in rows]
    fig, ax = plt.subplots(figsize=(7.1, 2.30))
    fig.subplots_adjust(left=.31, bottom=.25, right=.94, top=.97)
    y = list(range(len(rows)))
    ax.barh(y, values, height=.57, color=['#597c9d', '#648b7c', '#a58968', '#7e8792'])
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel('Correct validation questions lost relative to 758 / 798', fontsize=9)
    ax.tick_params(axis='both', length=0, labelsize=9, pad=6)
    ax.xaxis.grid(True, color='#e2e5e8', linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for at, value in zip(y, values):
        ax.text(value + .25, at, f'−{value}', va='center', fontsize=9)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(FIGURES / f'validation_ablation.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def setup_fonts():
    font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    fonts = {'Body': 'DejaVuSerif.ttf', 'BodyBold': 'DejaVuSerif-Bold.ttf',
             'BodyItalic': 'DejaVuSerif-Italic.ttf', 'Sans': 'DejaVuSans.ttf',
             'SansBold': 'DejaVuSans-Bold.ttf', 'Mono': 'DejaVuSansMono.ttf'}
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='SansBold', italic='Sans', boldItalic='SansBold')


def make_styles():
    base = dict(fontName='Body', fontSize=10.15, leading=14.0, textColor=colors.black,
                spaceAfter=7, allowWidows=0, allowOrphans=0)
    return {
        'body': ParagraphStyle('BodyText', **base),
        'small': ParagraphStyle('Small', **{**base, 'fontSize': 8.7, 'leading': 11.7, 'spaceAfter': 6}),
        'caption': ParagraphStyle('Caption', **{**base, 'fontName': 'Sans', 'fontSize': 8.4, 'leading': 11.1, 'spaceAfter': 10}),
        'h1': ParagraphStyle('Heading1', fontName='BodyBold', fontSize=14.2, leading=18.5,
                             textColor=colors.black, spaceBefore=2, spaceAfter=10, keepWithNext=True),
        'h2': ParagraphStyle('Heading2', fontName='BodyBold', fontSize=11.0, leading=14.2,
                             textColor=colors.black, spaceBefore=6, spaceAfter=6, keepWithNext=True),
        'title': ParagraphStyle('Title', fontName='BodyBold', fontSize=20, leading=25,
                                textColor=colors.black, spaceAfter=10),
        'subtitle': ParagraphStyle('Subtitle', fontName='Sans', fontSize=9.1, leading=12.5,
                                   textColor=colors.black, spaceAfter=17),
        'table': ParagraphStyle('TableBody', fontName='Sans', fontSize=8.7, leading=11.6, spaceAfter=0),
        'tablehead': ParagraphStyle('TableHead', fontName='SansBold', fontSize=8.7, leading=11.6, spaceAfter=0),
        'tablecenter': ParagraphStyle('TableCenter', fontName='Sans', fontSize=8.7, leading=11.6, alignment=TA_CENTER, spaceAfter=0),
        'tableheadcenter': ParagraphStyle('TableHeadCenter', fontName='SansBold', fontSize=8.7, leading=11.6, alignment=TA_CENTER, spaceAfter=0),
        'code': ParagraphStyle('Code', fontName='Mono', fontSize=7.65, leading=10.5, spaceBefore=3, spaceAfter=8),
        'equation': ParagraphStyle('Equation', **{**base, 'fontSize': 9.5, 'leading': 14.0, 'leftIndent': 12, 'spaceAfter': 7}),
    }


def build(author='', student_id=''):
    FIGURES.mkdir(exist_ok=True)
    metrics = read(HERE / 'data/metrics.json')
    hardware = read(HERE / 'data/hardware.json')
    run = CODE / 'artifacts/v73'
    experiment = read(run / 'experiments.json')
    local = read(CODE / 'runs/v73-reference-study/local-test/test-benchmark.json')
    remote = read(CODE / 'runs/v73-reference-study/clean-new-ser-test/test-benchmark.json')
    freeze = read(run / 'freeze.json')
    pipeline_figure()
    ablation_figure(metrics)
    setup_fonts()
    styles = make_styles()
    pages = []

    def p(text, kind='body'):
        return Paragraph(text, styles[kind])

    def table(headers, rows, widths, font=None):
        numeric_columns = {i for i, header in enumerate(headers) if i and header not in ('Use', 'Purpose', 'Interpretation')}
        data = [[p(html.escape(str(cell)), 'tableheadcenter' if i in numeric_columns else 'tablehead') for i, cell in enumerate(headers)]]
        data += [[p(html.escape(str(cell)), 'tablecenter' if i in numeric_columns else 'table') for i, cell in enumerate(row)] for row in rows]
        result = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
        style = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9edf1')),
                 ('GRID', (0, 0), (-1, -1), .45, colors.HexColor('#d9d9d9')),
                 ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                 ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                 ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]
        result.setStyle(TableStyle(style))
        result.spaceAfter = 9
        return result

    def image(name, width=CONTENT_WIDTH):
        from PIL import Image as PILImage
        path = FIGURES / name
        with PILImage.open(path) as im:
            height = width * im.height / im.width
        return Image(str(path), width=width, height=height)

    def page(items):
        height = 0
        for item in items:
            _, h = item.wrap(CONTENT_WIDTH, CONTENT_HEIGHT)
            height += h + item.getSpaceBefore() + item.getSpaceAfter()
        print(f'Planned page {len(pages) + 1}: approximately {height:.1f} / {CONTENT_HEIGHT:.1f} pt')
        pages.append(items)

    # Page 1
    identity = 'DASE7506 Advanced Machine Learning Methods | MP2 | 31 October 2026'
    if author:
        identity += '<br/>' + html.escape(author)
        if student_id:
            identity += ' | ' + html.escape(student_id)
    elif student_id:
        identity += '<br/>Student ID ' + html.escape(student_id)
    page([
        p('Selecting Reliable Reasoning<br/>with Arithmetic and Candidate Attention', 'title'),
        p(identity, 'subtitle'),
        p('Abstract', 'h2'),
        p('I study whether arithmetic evidence and candidate interactions can improve selection among four fixed mathematical solutions under a small deployment budget. '
          'The submitted method combines a multi-view ranking and tree branch with two compact self-attention families, then averages scores for numerically equivalent answers. '
          'Arithmetic consistency, question context and candidate interactions provide complementary evidence for a compact learned verifier. '
          'The frozen model selects correct final answers on <b>402 of 467 test questions, or 86.0814%</b>, with identical predictions on two hosts. '
          'Complete CPU prediction and independent scoring take 8.893 seconds on an AMD EPYC 9354 using four numerical threads; inference files total 6.5046 MiB. '
          'Validation-only component deletions support the combination, while failure cases show that arithmetic features do not guarantee semantic correctness. '
          'Validation was used adaptively. The test result is benchmark reproduction rather than a blind evaluation, and the specified reference Xeon has not been measured.'),
        p('1 Task and evaluation setting', 'h1'),
        p('The task is answer selection, rather than generating or repairing an answer. Each question has four supplied reasoning traces and a fixed 1024-dimensional embedding for each trace. '
          'I return one candidate index. The official parser maps equivalent fractions and decimals to the same numeric answer; the scorer checks this final answer, not the validity of every reasoning step. '
          'The data derive from GSM8K [1], but all experiments here use the course bank and its fixed representations.'),
        p('<b>Table 1.</b> Official splits and their experimental roles.', 'caption'),
        table(['Split', 'Questions', 'Informative', 'Use'], [
            ['TRAIN', '4,755', '2,114', 'Fit parameters and statistics'],
            ['Validation', '798', '355', 'Development and fusion selection'],
            ['TEST', '467', '467', 'Frozen scoring and reproduction'],
        ], [77, 76, 84, CONTENT_WIDTH - 237]),
        p('An informative group contains at least two distinct valid parsed answers. The remaining 443 validation groups have one shared valid answer, so their inclusion makes overall validation accuracy easier than the all-informative test bank. '
          'I therefore report both overall and informative validation accuracy. Multiple correct candidates are allowed; labels are obtained only from the corresponding permitted reference split.'),
        p('The research question is whether complementary evidence improves selection beyond answer frequency and individual learned components while keeping complete CPU evaluation below the time and asset limits. '
          'I evaluate the implementation through individual components, fixed-weight ablations and cross-host reproduction.'),
    ])

    # Page 2
    page([
        p('2 Pipeline and evidence construction', 'h1'),
        p('Figure 1 separates fitting from deployment. All learned normalizers, projections, lexical statistics and model parameters come from TRAIN. '
          'Validation chooses a frozen fusion configuration. During inference the method reads only the current question, candidates, supplied embeddings and checkpoint.'),
        image('pipeline.png'),
        p('<b>Figure 1.</b> The submitted pipeline. OOF scores are predictions for TRAIN questions held out of their base-model fit. '
          'Only the selected Logloss tree, five balanced compact attention members and three rich attention members are deployed; the five linear margins are inputs to the tree.', 'caption'),
        p('<b>Table 2.</b> Feature families and tree input sizes.', 'caption'),
        table(['Evidence', 'Dimension', 'Purpose'], [
            ['Rich candidate features', '207', 'Format, answer votes, arithmetic and relative evidence'],
            ['Compact candidate features', '22', 'Low-dimensional input for compact attention'],
            ['Repair features', '39', 'Numeric error propagation and answer support'],
            ['Quantity and schema features', '126 + 126', 'Compatibility with question quantities and forms'],
            ['Tree inputs', '619 / 788', 'OOF, PCA, quantity and extended evidence'],
        ], [164, 70, CONTENT_WIDTH - 234]),
        p('The arithmetic module parses bounded numeric expressions and evaluates them with exact fractions. It checks explicit equalities and follows repeated erroneous numbers through later calculations. '
          'Number coverage, simple algebraic schemas and repair support become features, rather than commands to override the model. Matching repeated numbers is heuristic: equal values can have different roles in a word problem.'),
        p('The 619-dimensional table contains 207 rich features, 30 base-score features, 256 PCA-derived context features and 126 quantity features. '
          'The extended table adds 126 schema, four lexical and 39 repair columns, giving 788. The tree combines these semantic, lexical and numeric signals in a shared per-candidate feature space.'),
    ])

    # Page 3
    page([
        p('3 Training objectives and model selection', 'h1'),
        p('Five linear rankers use the original candidate embedding, its question context, or its same-answer cluster context, together with rich features. '
          'Two views also use derived groups with one positive and three negatives, or three positives and one negative. '
          'Derived rows remain attached to their original question. Each fitting question has equal total pair mass; duplicates of the same answer do not multiply that answer cluster\'s weight.'),
        p('Five stratified question folds (seed 7526) produce base and lexical OOF evidence for stacked generalization [5]. Normalizers are estimated from fitting questions only. '
          'PCA64 and the final TF-IDF statistics are fitted on full TRAIN. The Logloss CatBoost head [4] uses a TRAIN-only 80/20 informative-question holdout, seed 7546, to choose iterations, then refits on all 2,114 informative TRAIN questions. '
          'The deployed head has 309 depth-six trees.'),
        p('3.1 Joint candidate scorers', 'h2'),
        p('The attention scorers compare the four candidates jointly using pre-normalized self-attention [2], without positional embeddings. '
          'This is related to attention-based set modelling [3], but does not implement inducing-point attention or a Set Transformer decoder. '
          'The compact input is 1024 raw coordinates plus 22 features. The rich input adds a 1024-dimensional question residual, 207 features and 39 repair features, totalling 2294.'),
        p('<b>Table 3.</b> Neural training configurations. LR is the AdamW learning rate.', 'caption'),
        table(['Family', 'Width / layers', 'Members', 'Epochs', 'LR / decay'], [
            ['Compact balanced', '128 / 2', '5', '15', '0.0001 / 0.001'],
            ['Rich balanced', '96 / 1', '3', '1', '0.0003 / 0.08'],
        ], [125, 91, 60, 55, CONTENT_WIDTH - 331]),
        p('Every scorer has four attention heads and an expansion factor of two. Compact models use GELU, dropout 0.2 and final layer normalization; rich models use ReLU, dropout 0.3 and a zero-initialized linear skip. '
          'Compact members fit four fifths of TRAIN using a seed-13 question permutation and seed 37 initialization. Rich members fit all TRAIN with seeds 7506, 7507 and 7508. '
          'AdamW uses batches of at most 64 and gradient clipping at 1.0.'),
        p('For scores s and sets V of valid candidates and P of correct candidates, the multi-positive mass loss is', 'body'),
        p('L<sub>mass</sub> = log ∑<sub>j∈V</sub> exp(s<sub>j</sub>) − log ∑<sub>j∈P</sub> exp(s<sub>j</sub>).', 'equation'),
        p('Compact training uses (1 + hard) × (BCE + 0.5 L<sub>mass</sub>). Rich training uses 0.5 L<sub>pair</sub> + 0.5 L<sub>mass</sub> + 0.15 BCE, '
          'where L<sub>pair</sub> averages softplus(s<sub>negative</sub> − s<sub>positive</sub>) over eligible pairs. These losses preserve multiple correct answers.'),
        p('3.2 Frozen fusion', 'h2'),
        p('A prescribed 309-configuration validation grid compares weights in sixths and pooling strengths 0, 0.6 and 1. The tree weight total is restricted to one third through two thirds, with a nonzero compact-family contribution. '
          'Selection prioritizes correct questions, then correct choices on wrong-majority single-correct groups, then candidate Brier loss. The selected probabilities are p = 0.5 T + (1/3) C + (1/6) R. '
          'Within each parsed-answer group they are averaged; the highest group score determines the first candidate with that answer. FP16-exported parameters are used before validation selection.'),
    ])

    # Page 4
    names = {'tree_log': 'Logloss tree', 'compact_balanced': 'Balanced compact attention',
             'rich': 'Rich attention'}
    result_rows = []
    deployed_components = [row for row in metrics['components'] if row['name'] in names]
    for row in metrics['baseline'] + deployed_components + [metrics['full']]:
        label = names.get(row['name'], row['name'])
        result_rows.append([label, f"{row['correct']}/798", f"{row['accuracy']:.2f}%",
                            f"{row['informative_correct']}/355", f"{row['trap_correct']}/60"])
    page([
        p('4 Validation and formal test results', 'h1'),
        p('Table 4 compares answer-frequency and first-candidate baselines with the three deployed components and the frozen ensemble. '
          'All learned component rows use same-answer mean pooling. The 60 trap questions have exactly one correct candidate and a repeated wrong answer that wins the frequency baseline.'),
        p('<b>Table 4.</b> Validation results. The last two columns separate informative questions and wrong-majority traps.', 'caption'),
        table(['Method', 'Correct', 'Overall', 'Informative', 'Traps'], result_rows,
              [192, 75, 74, 83, CONTENT_WIDTH - 424]),
        p('The ensemble gains 70 correct validation choices over majority voting, with 82 fixes and 12 regressions. On the informative subset it reaches 315/355, or 88.73%, compared with 245/355, or 69.01%, for majority. '
          'Relative to the Logloss tree it fixes nine choices and loses two, a net gain of seven. The rich family is strongest on the trap subset at 47/60, while the ensemble has 41/60: optimizing total accuracy does not maximize every subgroup.'),
        p('4.1 Formal test scoring', 'h2'),
        p('The fusion configuration was selected on validation and frozen before test scoring. The official result is <b>402/467 = 86.0813704496788%</b>. '
          'The local workstation and new-ser produce identical IDs and all 467 choices, and the standalone release reproduces them. '
          'Component and ablation TEST cells are not evaluated. The 94.99% validation and 86.08% test percentages have different question compositions and should not be compared as if drawn from the same mix.'),
        p('Fitting uses TRAIN, selection uses validation, and TEST is used for frozen scoring and reproduction. '
          'The test result should be interpreted as benchmark reproduction rather than an independent blind evaluation. '
          'No TEST item-level correctness analysis is performed.'),
    ])

    # Page 5
    ablation_labels = ['Remove Logloss tree', 'Remove balanced compact attention', 'Remove rich attention',
                       'Remove same-answer pooling']
    ablation_rows = []
    for label, row in zip(ablation_labels, metrics['ablations']):
        ablation_rows.append([label, f"{row['correct']}/798", f"{row['delta_correct_vs_full']:+d}",
                              f"{row['trap_correct']}/60"])
    page([
        p('5 Controlled validation diagnostics', 'h1'),
        p('These diagnostics were added after final test scoring to explain the submitted frozen model. They do not change it. '
          'Deleting a family renormalizes the remaining weights without a new weight search or retraining. The pooling ablation retains all probabilities and weights.'),
        p('<b>Table 5.</b> Fixed component and pooling ablations. Differences are correct questions relative to 758/798.', 'caption'),
        table(['Change', 'Validation', 'Difference', 'Traps'], ablation_rows,
              [268, 86, 78, CONTENT_WIDTH - 432]),
        image('validation_ablation.png'),
        p('<b>Figure 2.</b> Losses in correct validation choices after each fixed change. The component-deletion results include renormalization, not reoptimized reduced ensembles.', 'caption'),
        p('The tree contributes the largest aggregate benefit: removing it loses 14 correct choices. Removing the balanced compact and rich families loses five and three. '
          'Pooling adds a net five choices, consistent with reducing reliance on one unusually high-scoring trace for a repeated answer.'),
        p('These are development-set diagnostics, not independent confirmation. Deleting a component also changes score calibration and its normalized mixture weights. '
          'The results do not identify the isolated causal benefit of each feature or normalization choice. '
          'Those details would need matched refits with a separately specified validation protocol.'),
    ])

    # Page 6
    page([
        p('6 Validation examples and remaining failures', 'h1'),
        p('Both examples below are drawn from validation. They illustrate behaviour rather than estimate how often each failure type occurs; exact validation IDs, traces and probabilities are included in the report data.'),
        p('6.1 Pooling can overcome a confident wrong trace', 'h2'),
        p('A show has ten seasons, with twenty episodes per season for the first half and twenty-five for the second half. The correct total is 5 × 20 + 5 × 25 = 225. '
          'Three candidates instead apply each episode count to all ten seasons and answer 450. Their individual multiplications are valid; the mistake is the interpretation of “half”.'),
        p('<b>Table 6.</b> Candidate and answer-level scores for the show example.', 'caption'),
        table(['Answer', 'Candidates', 'Highest candidate score', 'Mean answer score'], [
            ['225 correct', '1', '0.67229', '0.67229'],
            ['450 wrong', '3', '0.75817', '0.66494'],
        ], [106, 84, 155, CONTENT_WIDTH - 345]),
        p('Selecting the highest candidate probability chooses 450, while answer pooling chooses 225. The answer-score gap is only 0.00735, so this success is also an example of a fragile decision. '
          'Arithmetic checking alone does not resolve the mistaken allocation of seasons.'),
        p('6.2 Repeated arithmetic errors can still win', 'h2'),
        p('A population question requires (15 × 58,940 − 130,000)/2 = 377,050. Three traces instead write 15 × 58,940 = 883,100, propagate the 1,000-person error and answer 376,550. '
          'The exact arithmetic extractor detects this multiplication discrepancy in all three wrong traces, but their mean score is 0.65004 versus 0.57592 for the correct trace. '
          'The ensemble still selects the wrong answer. Verification signals enter a learned score; they do not impose a formal correctness constraint.'),
        p('6.3 Score gaps and uncertainty', 'h2'),
        p('<b>Table 7.</b> Validation accuracy by gap between the two highest answer scores.', 'caption'),
        table(['Top two answer-score gap', 'Informative questions', 'Correct', 'Accuracy'], [
            ['Below 0.25', '54', '35', '64.81%'],
            ['At least 0.25', '301', '280', '93.02%'],
        ], [189, 125, 80, CONTENT_WIDTH - 394]),
        p('Small gaps identify a difficult subset on validation, but they are not calibrated probabilities of correctness. '
          'The first example has a very small gap and is correct, while the population example is wrong at a gap of 0.07412. '
          'No threshold is used to change the frozen test decisions.'),
        p('6.4 Scope of the evidence', 'h2'),
        p('The validation set has been reused across development. The tree iteration holdout uses a fixed OOF base table and is not fully nested cross-validation of the entire pipeline. '
          'Numeric matching can confuse repeated values with different roles and miss semantic constraints. '
          'Separate audit-bank generalization and runtime on the reference Xeon remain unmeasured.'),
    ])

    # Page 7
    asset_rows = []
    for path in sorted((run / 'checkpoint').iterdir()):
        asset_rows.append([path.name, f'{path.stat().st_size:,}', f'{path.stat().st_size / 2**20:.4f}'])
    page([
        p('7 Computation and resource declaration', 'h1'),
        p('I trained the submitted v73 on my local workstation with an Intel Core i9-14900KF, 62.58 GiB of OS-visible RAM and <b>one NVIDIA GeForce RTX 4090D</b> '
          '(24,564 MiB reported GPU memory, driver 580.173.02). CPU numerical libraries were configured for four threads. Neural members were fitted sequentially on that GPU; no multi-GPU training was used.'),
        p(f'The recorded final v73 run took <b>{experiment["seconds"]:.2f} seconds</b>, including feature extraction, fitting, validation and the prescribed fusion search. '
          f'Peak PyTorch allocated GPU memory was {experiment["peak_gpu_bytes"] / 2**20:.2f} MiB; this counter excludes non-PyTorch allocations and is not total system GPU memory. '
          'The recorded runtime excludes environment setup, audits and report production.'),
        p('7.1 Complete fresh-process CPU evaluation', 'h2'),
        p('<b>Table 8.</b> Complete official test timing with four numerical threads.', 'caption'),
        table(['Host CPU', 'Prediction', 'Scoring', 'Total'], [
            ['Intel Core i9-14900KF', f'{local["prediction_wall_seconds"]:.3f} s', f'{local["scoring_wall_seconds"]:.3f} s', f'{local["total_wall_seconds"]:.3f} s'],
            ['AMD EPYC 9354 new-ser', f'{remote["prediction_wall_seconds"]:.3f} s', f'{remote["scoring_wall_seconds"]:.3f} s', f'{remote["total_wall_seconds"]:.3f} s'],
        ], [226, 91, 88, CONTENT_WIDTH - 405]),
        p('Inference uses CPU NumPy only with four numerical threads. The measurement starts before each fresh prediction/scoring process and includes imports, supplied-input loading, model loading, feature calculation and every deployed member. '
          'A separate validation process inspection observed a peak of four OS threads, no child workers and no loaded CUDA or Torch libraries. Parsing caches start empty in the fresh process and store only computations on current inputs.'),
        p('The new-ser server has 125.77 GiB OS-visible RAM. Neither reported CPU is the specified Xeon Platinum 8457C, so the 30-second reference-machine qualification is not claimed as verified.'),
        p('7.2 All deployed inference assets', 'h2'),
        p('<b>Table 9.</b> All files in the deployed checkpoint, without outer-package compression.', 'caption'),
        table(['Checkpoint file', 'Bytes', 'MiB'], asset_rows + [['Total', f'{freeze["inference_asset_bytes"]:,}', f'{freeze["inference_asset_bytes"] / 2**20:.4f}']],
              [265, 134, CONTENT_WIDTH - 399]),
        p('The NPZ entries are uncompressed. FP16 neural parameters are converted to FP32 for CPU computation; even counting all arrays after this expansion gives 11.9954 MiB, below 16 MiB before the small container/configuration overhead. '
          'All deployed normalizers, projections, lexical coefficients, configuration and ensemble members are included. Training-only diagnostic caches are not read by inference. '
          'No external dataset, additional pretrained weights, evaluation-time generation, network API or stored-answer lookup is used.'),
    ])

    # Page 8
    page([
        p('8 Reproduction and related work', 'h1'),
        p('The release keeps the official data, parser and scorer unchanged. The entry point is <font name="Mono">code/submission_v73.py</font>; '
          'the matching assets are in <font name="Mono">code/artifacts/v73/checkpoint</font>. Frozen hashes, the before-test selection record, same-host remote receipt, '
          'official score JSON and cross-host choice comparisons are included. Source and asset hashes verify the submitted implementation.'),
        p('8.1 Reproduction commands', 'h2'),
        p('From <font name="Mono">code/</font>, use Python 3.12 and the pinned NumPy-only inference requirements. '
          'The verifier sets the four-thread limit and common numerical backend before launching the official tools. Output directories must be new.'),
        Preformatted('python -m pip install -r requirements-inference-v73.txt\n'
                     'python -m reference_rebuild_v73.verify --run artifacts/v73 \\\n  --split test --output runs/reproduced-test\n'
                     'python report_analysis.py --output-dir ../report/recomputed-data', styles['code']),
        p('Training uses a separate Python 3.13 CUDA environment with pinned NumPy, SciPy, scikit-learn, CatBoost and PyTorch versions. '
          'The supplied training script needs no previous checkpoint or cache. To replay the full training audit, use the newly generated training output, since bulky search caches are omitted from the submission package.'),
        Preformatted('python -m pip install -r requirements-train-v73.txt\n'
                     'CUDA_VISIBLE_DEVICES=0 python -m reference_rebuild_v73.train \\\n  --output artifacts/retrained-v73\n'
                     'python -m unittest test_reference_rebuild_v73 -v', styles['code']),
        p('9 unit tests pass. Normalization and validation-component replay errors were zero; base OOF probability differences were below 3 × 10<super>−8</super>. '
          'Sampled unknown IDs, candidate/question permutations, single-question calls and blocked reference I/O passed. Cross-hardware retraining is not promised to be bitwise identical; frozen assets reproduce the submitted score.', 'small'),
        p('8.2 Related work', 'h2'),
        p('Verifier-based selection [1] provides an outcome-supervision perspective for scoring the four fixed candidates. '
          'Candidate interactions use self-attention [2] over a set of solutions [3]. '
          'The tabular branch uses CatBoost [4] for nonlinear feature interactions. '
          'Held-question base predictions provide inputs to its meta model, following the principle of stacked generalization [5].', 'small'),
        p('8.3 AI assistance', 'h2'),
        p('<b>OpenAI Codex substantially assisted implementation, experiments, diagnostics, figures and report drafting.</b> '
          'Saved experiment records and frozen assets accompany the code for reproduction.', 'small'),
        p('References', 'h2'),
        p('[1] K. Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021. '
          '<link href="https://arxiv.org/abs/2110.14168" color="#000000">https://arxiv.org/abs/2110.14168</link>', 'small'),
        p('[2] A. Vaswani et al. Attention Is All You Need. Advances in Neural Information Processing Systems 30, 2017. '
          '<link href="https://arxiv.org/abs/1706.03762" color="#000000">https://arxiv.org/abs/1706.03762</link>', 'small'),
        p('[3] J. Lee et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML, PMLR 97, pp. 3744-3753, 2019. '
          '<link href="https://proceedings.mlr.press/v97/lee19d.html" color="#000000">https://proceedings.mlr.press/v97/lee19d.html</link>', 'small'),
        p('[4] L. Prokhorenkova et al. CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems 31, 2018. '
          '<link href="https://arxiv.org/abs/1706.09516" color="#000000">https://arxiv.org/abs/1706.09516</link>', 'small'),
        p('[5] D. H. Wolpert. Stacked generalization. Neural Networks 5(2), pp. 241-259, 1992. '
          '<link href="https://doi.org/10.1016/S0893-6080(05)80023-1" color="#000000">https://doi.org/10.1016/S0893-6080(05)80023-1</link>', 'small'),
    ])

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.black)
        canvas.setFont('Sans', 7.5)
        canvas.drawString(MARGIN, 27, 'DASE7506 MP2   |   Frozen submission v73')
        canvas.drawRightString(WIDTH - MARGIN, 27, str(doc.page))
        canvas.restoreState()

    story = []
    for i, items in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.extend(items)
    path = HERE / 'MP2_Report.pdf'
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                 topMargin=44, bottomMargin=44, title='Selecting Reliable Reasoning with Arithmetic and Candidate Attention',
                                 author=author, subject='DASE7506 MP2 frozen v73 method experiments and computation declaration',
                                 pageCompression=1)
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--author', default='')
    parser.add_argument('--student-id', default='')
    options = parser.parse_args()
    build(options.author, options.student_id)
