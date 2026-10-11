import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('AuditSans','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('AuditSansBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('AuditMono','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))

root=Path('dinn-project')
m=json.loads((root/'audit/results/metrics.json').read_text())
repeat=json.loads((root/'audit/repeat-results/metrics.json').read_text())
compare_keys=['recovered_parameters','relative_errors_fraction','final_losses','trajectory_rmse_by_compartment',
    'equation_residual_rmse_per_day_by_compartment','population_conservation_max_abs_error',
    'minimum_by_compartment','maximum_by_compartment','initial_condition_max_abs_error']
repro={key:m[key]==repeat[key] for key in compare_keys}
comparison={'same_seed':True,'same_recorded_environment':m['environment']==repeat['environment'],
    'all_compared_metrics_identical':all(repro.values()),'comparisons':repro,
    'first_run_utc':m['utc_time'],'repeat_run_utc':repeat['utc_time']}
(root/'audit/results/repeatability.json').write_text(json.dumps(comparison,indent=2))

out=Path('output/pdf');out.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='MainTitle',fontName='AuditSansBold',fontSize=19,leading=25,spaceAfter=12,textColor=colors.black))
styles.add(ParagraphStyle(name='Section',fontName='AuditSansBold',fontSize=11,leading=15,spaceBefore=12,spaceAfter=7,textColor=colors.black))
styles.add(ParagraphStyle(name='BodyAudit',fontName='AuditSans',fontSize=9.5,leading=13.5,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallAudit',fontName='AuditSans',fontSize=8.2,leading=10.5,spaceAfter=6))
styles.add(ParagraphStyle(name='CellAudit',fontName='AuditSans',fontSize=8.5,leading=11.5))
styles.add(ParagraphStyle(name='CodeAudit',fontName='AuditMono',fontSize=8,leading=11,spaceAfter=7))
story=[]
def p(text,style='BodyAudit'):story.append(Paragraph(text,styles[style]))
def table(rows,widths):
    cells=[[Paragraph(str(v),styles['CellAudit']) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),
        ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(t);story.append(Spacer(1,9))

p('SIRD DINN Reproduction Audit','MainTitle')
p('Step 1 | Kingsley Chimuanya Ukandu | October 11, 2026','SmallAudit')
p('The existing synthetic SIRD implementation reproduces parameter recovery below 0.1 percent relative error for all three rates in the recorded environment. A second fresh run gives identical compared metrics. The network also produces negative compartment values and population-balance deviations, so the result supports a controlled parameter-estimation proof of implementation, with explicit numerical limitations.')
p('Scope and provenance','Section')
p('Audited source: kingsleyukandu/dinn-epidemic-mobility, notebooks/sird-dinn.ipynb. The original source notebook was retained without changes. All ordinary Python cells were executed sequentially in a fresh Python process; old outputs and interactive state were not used. Passive loss logging and post-training diagnostics were added to an audit copy.')
p('Source commit: '+m['source_commit'],'CodeAudit')
p('Recovered parameters','Section')
rows=[['Rate','True value','Historical display','Fresh estimate','Relative error %']]
hist={'beta':'0.5000','gamma':'0.2000','omega':'0.0715'}
for k in m['true_parameters']:
    rows.append([k,f"{m['true_parameters'][k]:.9f}",hist[k],f"{m['recovered_parameters'][k]:.9f}",f"{m['relative_errors_percent'][k]:.6f}"])
table(rows,[58,79,90,96,96])
p('Historical values are the rounded values saved in the original notebook, not full-precision estimates. The fresh transmission estimate rounds to 0.5002, while the original display is 0.5000. The broader claim that all three reported relative errors are below 0.1 percent is reproduced, but the original rounded values are not all reproduced exactly. The mortality error is approximately 0.0965 percent, close to the stated threshold.','SmallAudit')
p('Relative error = abs(recovered - true) / abs(true). Percentage error = 100 x relative error. Four-decimal display alone is insufficient to evaluate the 0.1 percent threshold.','SmallAudit')

story.append(PageBreak())
p('Numerical Validity Diagnostics','MainTitle')
p('Diagnostics evaluate the final trained network on 641 points from day 0 to day 160. Population fractions sum to one in the reference SIRD system. Numerical checks describe this sampled grid; they do not prove properties at every time.','BodyAudit')
rows=[['Check','Measured result','Interpretation'],
['Positive rate parameters','All three rates positive','Enforced by exponential parameterization.'],
['Overall trajectory RMSE',f"{m['trajectory_rmse_overall']:.6g}",'Population-fraction error against the ODE reference.'],
['Population balance',f"Max |s+i+r+d-1| = {m['population_conservation_max_abs_error']:.6g}",f"Approximately {100*m['population_conservation_max_abs_error']:.4f}% of total population; conservation is not exact."],
['Compartment positivity',f"Minimum infected fraction = {m['minimum_by_compartment']['i']:.6g}",'Negative predictions occur; positivity is not enforced.'],
['Initial conditions',f"Max absolute error = {m['initial_condition_max_abs_error']:.6g}",'Initial conditions are penalized, not imposed exactly.'],
['Equation consistency',f"Max absolute residual = {m['equation_residual_max_abs_per_day']:.6g} per day",'Finite residual; equation satisfaction is approximate.'],
['ODE reference conservation',f"Max error = {m['independent_ode_grid_conservation_max_abs_error']:.3g}",'Reference integration preserves balance to numerical precision.']]
table(rows,[112,141,166])
p('Compartment specific checks','Section')
rows=[['State','Minimum fraction','Trajectory RMSE','Equation RMSE per day']]
for k in ['s','i','r','d']:
    rows.append([k.upper(),f"{m['minimum_by_compartment'][k]:.6g}",f"{m['trajectory_rmse_by_compartment'][k]:.6g}",f"{m['equation_residual_rmse_per_day_by_compartment'][k]:.6g}"])
table(rows,[72,111,111,125])
p(f"There are {m['negative_entries']} negative entries among 2,564 evaluated compartment values. Recovered and deceased trajectories also have negative derivatives at some sampled times. These deviations are small on the population-fraction scale, but can exceed the initial infected fraction of 0.000001. Good overall curve fit therefore does not establish biologically valid behavior near the start of the epidemic.",'SmallAudit')

story.append(PageBreak())
p('Trajectory and Training Results','MainTitle')
story.append(Image(str(root/'audit/results/trajectory_comparison.png'),width=6.0*inch,height=3.6*inch))
p('The training data are noiseless observations of all four compartments generated by the same SIRD equations used in the residual loss. Curves are fitted over the observed interval; this is not an independent forecast evaluation.','SmallAudit')
story.append(Image(str(root/'audit/results/training_diagnostics.png'),width=6.0*inch,height=2.28*inch))
p('Losses are logged immediately before each recorded Adam update. The equation loss above is expressed in normalized time tau = t / 160. Per-day equation diagnostics on the preceding page divide the derivative by 160. Loss fluctuations are retained, without selecting a favorable intermediate checkpoint.','SmallAudit')
p('Final losses after the last update','Section')
table([['Data MSE','Initial condition MSE','Equation MSE in tau','Weighted total'],
[f"{m['final_losses']['data_mse']:.5g}",f"{m['final_losses']['initial_condition_mse']:.5g}",f"{m['final_losses']['equation_residual_mse_tau_training_grid']:.5g}",f"{m['final_losses']['weighted_total']:.5g}"]],[104,104,107,104])

story.append(PageBreak())
p('Reproducibility and Next Actions','MainTitle')
p('Recorded configuration','Section')
c=m['configuration'];e=m['environment']
table([['Setting','Value'],['Execution','Fresh Python process; CPU; float32; one CPU thread'],
['Random seeds','PyTorch 0; NumPy 0'],['Network','Three hidden layers of width 32; tanh; four linear compartment outputs'],
['Training','Adam, learning rate 0.001; 20,001 updates (inclusive loop 0 to 20,000)'],
['Loss weights','Data 1; equations 0.0001; initial conditions 1'],
['Data and grids','54 observations every 3 days, 0 to 159; 161 collocation points, 0 to 160'],
['Environment',f"Python {e['python']}; PyTorch {e['torch']}; NumPy {e['numpy']}; SciPy {e['scipy']}; Matplotlib {e['matplotlib']}"],
['Repeatability','Two fresh runs: identical compared metrics' if comparison['all_compared_metrics_identical'] else 'Two fresh runs: differences recorded in repeatability.json']],[115,304])
p('The original notebook metadata reports Python 3.11.15 and does not provide a complete historical dependency manifest. The audit records a new environment; it cannot establish bit-for-bit agreement with the unspecified historical environment. Same-seed repeatability is not evidence of robustness across seeds, noise levels, observation patterns, or software versions.','SmallAudit')
p('Implications for the petition package','Section')
p('Retain the synthetic parameter-recovery proof of implementation, with the original historical results clearly distinguished from the fresh audit. When the petition is revised in Step 6, use verified full-precision values and describe training as penalizing deviations from the governing equations. Avoid implying exact equation satisfaction or guaranteed compartment positivity. The audit does not establish forecast superiority, real U.S. data validation, or completed mobility modeling.')
p('Recommended technical work after this audit','Section')
p('First address compartment positivity, population balance, and initial-condition accuracy through an explicit constraint design. Assess its effect on parameter recovery rather than assuming improvement. Then conduct the planned multiple-seed and observation-imperfection experiments and compare against conventional ODE calibration. These changes and experiments belong to subsequent authorized stages; they were not performed in Step 1.')
p('Reproduction files','Section')
p('The audit runner, sequentially executed notebook, full-precision metrics, parameter table, loss history, trajectory diagnostics, and dependency specifications have been prepared. Original model code and petition documents remain unchanged. GitHub publication and commits are reserved for the planned repository stage.','SmallAudit')

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D9D9D9'));canvas.line(54,42,558,42)
    canvas.setFont('AuditSans',7.5);canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(54,29,'Kingsley Chimuanya Ukandu | DINN Step 1 Audit | October 11, 2026')
    canvas.drawRightString(558,29,f'Page {doc.page}');canvas.restoreState()
SimpleDocTemplate(str(out/'DINN_Step1_Reproduction_Audit.pdf'),pagesize=letter,rightMargin=72,leftMargin=72,
    topMargin=54,bottomMargin=54,title='SIRD DINN Reproduction Audit',author='Kingsley Chimuanya Ukandu').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'report':str(out/'DINN_Step1_Reproduction_Audit.pdf'),'comparison':comparison},indent=2))
