"""Create the Step 3 report from verified final results, retaining all seeds."""
import json
from pathlib import Path
import zipfile

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path('dinn-project'); DATA = ROOT/'validation/step3/results'
rows = json.loads((DATA/'neural_summary.json').read_text())
protocol = json.loads((DATA/'protocol.json').read_text())
checks = json.loads((DATA/'model_checks.json').read_text())['results']
verification = json.loads((DATA/'verification.json').read_text())
assert verification['status'] == 'PASS'
baselines = {c:json.loads((DATA/c/'ode_baseline.json').read_text()) for c in ['clean','noisy','missing']}
names = ['beta1','gamma1','omega1','beta2','gamma2','omega2','tau12','tau21']
out = Path('output/pdf'); out.mkdir(parents=True,exist_ok=True)
for name,file in [('TSans','DejaVuSans.ttf'),('TBold','DejaVuSans-Bold.ttf'),('TMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
styles = getSampleStyleSheet()
for name,font,size,leading in [('TitleT','TBold',18.5,24),('HeadT','TBold',11,15),
    ('BodyT','TSans',9.2,13),('SmallT','TSans',8.2,11),('CellT','TSans',8,10.5),('MonoT','TMono',7.2,10)]:
    styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,spaceAfter=7))
story = []
def p(text,style='BodyT'): story.append(Paragraph(text,styles[style]))
def table(data,widths):
    t=Table([[Paragraph(str(v),styles['CellT']) for v in r] for r in data],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef2')),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d5dce2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,8)])
def page(title): story.append(PageBreak());p(title,'TitleT')
def subset(case): return [r for r in rows if r['case']==case]
def max_error(case,group): return max(r['relative_errors_percent'][n] for r in subset(case) for n in group)
def error_span(case,name):
    a=[r['relative_errors_percent'][name] for r in subset(case)]
    return f'{min(a):.4f} to {max(a):.4f}'
def small_error(value): return '<1e-9' if value<1e-9 else f'{value:.4f}'
better = all(baselines[c]['trajectory_rmse']<min(r['trajectory_rmse'] for r in subset(c)) for c in baselines)

p('Two-Region SIRD Mobility Validation','TitleT')
p('Step 3 | Kingsley Chimuanya Ukandu | October 11, 2026','SmallT')
p('A two-region SIRD DINN has been implemented and tested on synthetic observations. It estimates six region-specific disease rates and two directional movement rates. Across three clean-data training seeds, the largest relative rate error is '+f"{max_error('clean',names):.3f}%"+'. Nonnegative states, combined population balance, and exact initial conditions held in all nine final neural runs.')
p(('Conventional ODE calibration had lower overall trajectory RMSE in every tested condition. ' if better else 'The conventional ODE calibration results are retained for comparison. ')+
    'The extension supports an implemented synthetic mobility proof of concept. It does not establish real-data performance, uncertainty coverage, or superiority over conventional calibration. Neural equation residuals remain finite, and deceased curves can decrease slightly.')
p('Model and assumptions','HeadT')
table([['Item','Implemented choice'],
    ['Eight states','S1, I1, R1, D1, S2, I2, R2, D2; fractions of combined initial population'],
    ['Region sizes at day zero','Region 1: 0.6; region 2: 0.4'],
    ['Transmission','beta_j x S_j x I_j / p_j; p_j is the fixed initial regional population fraction'],
    ['Movement','tau12: region 1 to 2; tau21: region 2 to 1; identical coefficients for S, I, R'],
    ['Deceased states','No movement; cumulative deaths stay in their recorded region'],
    ['Unknowns','beta, gamma, omega in each region; both directional movement rates']],[117,351])
p('Fixed transmission denominators preserve the original single-region SIRD equations when movement is zero. They are a modeling assumption, not the current regional population. Local populations may change; they are not separately normalized to one. The combined accounted total includes living individuals and cumulative deaths.','SmallT')
p('Recorded experiment design','HeadT')
table([['Data condition','Observed timepoints','Neural seeds'],
    ['Clean; all eight states','54; days 0, 3, ..., 159','0, 1, 2'],
    ['Noisy; Gaussian sigma 0.001','54; one fixed noise realization','0, 1, 2'],
    ['Missing; 16 interior times removed','38; both endpoints retained','0, 1, 2']],[255,135,78])
p('All runs use the same known initial state, starting rates, network, loss, and optimizer budget. The protocol was recorded before validation runs. A separate development pilot is retained and excluded from the nine-run summary. Every final seed is reported; no true-error stopping or best-seed selection was used.','SmallT')

page('Equations and Limiting-Case Checks')
p('For j = 1, 2, the SIRD equations use the conservative movement generator L. Vectors S, I, and R each contain the two regional state fractions. The matrix uses the column convention: entry (destination, origin).','SmallT')
story.append(Image(str(DATA/'model_equations.png'),width=468,height=146))
table([['Movement generator L','Origin: region 1','Origin: region 2'],
    ['Destination: region 1','-tau12','tau21'],['Destination: region 2','tau12','-tau21']],[184,142,142])
p('Each column of L sums to zero. Summing the eight equations cancels infection, recovery, mortality transfer, and movement, giving d(sum of all states)/dt = 0. At any nonnegative boundary state, the derivative of a zero compartment is nonnegative. For the exact ODE, D_j is nondecreasing because its derivative is omega_j I_j. These properties differ from approximate equation satisfaction by a neural fit.','SmallT')
table([['Check','Recorded result'],
    ['Global and regional RHS balance',f"Largest algebraic discrepancy {max(checks['population_balance'].values()):.3g}"],
    ['Zero movement',f"Independent original SIRD equations: RHS discrepancy {checks['zero_mobility']['max_abs_rhs_difference_from_original_SIRD']:.3g}; trajectory discrepancy {checks['zero_mobility']['max_abs_trajectory_difference_from_independent_original_SIRD']:.3g}"],
    ['Transport without disease',f"Matrix-exponential agreement: max error {checks['transport_only']['max_abs_difference_from_matrix_exponential']:.3g}; deaths constant"],
    ['Identical symmetric regions',f"Aggregate single-SIRD agreement: max error {checks['symmetric_limit']['max_abs_aggregate_difference_from_single_SIRD']:.3g}"],
    ['Imported infection','Region 2 starts uninfected, becomes infected with movement; stays uninfected without movement'],
    ['Independent ODE solvers',f"DOP853 vs Radau: max difference {checks['independent_solver']['max_abs_DOP853_Radau_difference']:.3g}"],
    ['NumPy / PyTorch and derivatives','Matching RHS implementations; automatic derivatives checked by finite differences']],[160,308])
p('Numerical trajectory checks use an absolute tolerance of 1e-8 in combined-population fraction. Algebraic checks use much tighter roundoff tolerances. These are controlled implementation checks, not empirical validation.','SmallT')

page('Clean-Data Recovery and Comparator')
p('All rates below are per day. Each clean-data neural estimate is displayed; the error range spans all three seeds. Relative error (%) = 100 x abs(estimate - true rate) / abs(true rate). Full precision and each seed-specific error are retained in parameter_recovery.csv.','SmallT')
table([['Rate','True','Seed 0','Seed 1','Seed 2','Error range %']]+[
    [n,f"{protocol['synthetic_truth'][n]:.7f}",*[f"{next(r for r in subset('clean') if r['seed']==s)['parameters'][n]:.7f}" for s in [0,1,2]],error_span('clean',n)]
    for n in names],[55,76,76,76,76,109])
p('Conventional ODE calibration','HeadT')
table([['Data','Max disease error %','Max movement error %','Trajectory RMSE']]+[
    [c.capitalize(),small_error(max(baselines[c]['relative_errors_percent'][n] for n in names[:6])),
     small_error(max(baselines[c]['relative_errors_percent'][n] for n in names[6:])),f"{baselines[c]['trajectory_rmse']:.3g}"]
    for c in ['clean','noisy','missing']],[70,134,140,124])
p('The comparator fits the same observations, known initial state, and starting rate guesses by least squares on log rates. All three fits converged successfully. Synthetic data and calibration use the same equations and DOP853 integration, a favorable model-correct setting for ODE calibration. Bounds are 0.0001 to 3 per day for disease rates and 0.00001 to 0.5 for movement rates.','SmallT')
story.append(Image(str(DATA/'mobility_recovery.png'),width=468,height=157))
p('The figure shows every seed for both movement directions under each condition, alongside the ODE comparator. Good recovery in this dataset does not establish identifiable mobility under other regional configurations.','SmallT')

page('Noise, Missingness, and Seed Sensitivity')
p('Noise standard deviation is 0.001 of the combined initial population, independently applied to each state. Negative measurements are retained, without clipping or renormalization. The missing-data experiment removes 16 of 54 observation times (29.6%) using a separate fixed random seed. Each condition uses one dataset and three neural initializations.','SmallT')
table([['Rate','Noisy error range %','Missing error range %']]+[[n,error_span('noisy',n),error_span('missing',n)] for n in names],[84,192,192])
p('All final neural runs','HeadT')
table([['Data','Seed','Max disease error %','tau12 error %','tau21 error %','State RMSE']]+[
    [r['case'].capitalize(),r['seed'],f"{max(r['relative_errors_percent'][n] for n in names[:6]):.4f}",
     f"{r['relative_errors_percent']['tau12']:.4f}",f"{r['relative_errors_percent']['tau21']:.4f}",f"{r['trajectory_rmse']:.3g}"]
    for r in rows],[65,35,100,91,91,86])
p('These error ranges describe the retained experiments, not statistical confidence intervals. There are no independent repetitions of noise or missingness, no practical observation model, and no partial-state recovery experiment. Known initial conditions and observations of all eight states make the problem easier than typical surveillance-based estimation.','SmallT')

page('Trajectories and Physical-State Diagnostics')
p('All three clean-data neural seeds are plotted against the ODE reference. State values use the combined initial population as denominator. Regional population is the sum of the four states recorded in that region, including cumulative deaths.','SmallT')
story.append(Image(str(DATA/'clean_trajectories.png'),width=468,height=250))
story.append(Image(str(DATA/'regional_population.png'),width=468,height=130))
table([['Diagnostic across all nine final neural runs','Measured result'],
    ['Negative state entries on 641-point grids',sum(r['negative_entries'] for r in rows)],
    ['Largest combined population-balance error',f"{max(r['conservation_max_abs_error'] for r in rows):.3g}"],
    ['Largest initial-state error',f"{max(r['initial_max_abs_error'] for r in rows):.3g}"],
    ['Largest per-day ODE residual',f"{max(r['residual_max_abs_per_day'] for r in rows):.3g}"],
    ['Largest regional-balance residual per day',f"{max(r['regional_balance_residual_max_abs_per_day'] for r in rows):.3g}"]],[321,147])
count=sum(r['negative_D_derivative_entries']>0 for r in rows)
p(f'Nonnegative states, combined conservation, and the initial state are imposed by the simplex blend. Equation consistency and local population-balance identities are penalized, rather than imposed exactly. Small decreases in deceased states occur in {count} of the nine neural runs; the most negative measured derivative is {min(r["minimum_D_derivative_per_day"] for r in rows):.3g} per day. Regional recovered counts may legitimately decrease through outward movement.','SmallT')

page('Interpretation and Reproduction')
p('What this stage establishes','HeadT')
p('The project now contains a separate eight-state, eight-rate two-region SIRD implementation, conservative directional movement equations, synthetic parameter-recovery experiments across three seeds, noise and missingness checks, and a conventional ODE comparator. The original single-region notebook and earlier validation records are preserved. These new results date to October 11, 2026; they do not retroactively describe the earlier public repository.')
p('Identifiability limit','HeadT')
p('With equal initial regional populations, identical disease rates and identical compartment trajectories, symmetric movement generates no net flow. In the recorded check, changing both movement rates from 0.005 to 0.03 per day leaves trajectories unchanged. The common movement scale is therefore unidentifiable in that symmetric setup, even with all compartments observed.')
p('For the asymmetric calibration data, the observed-state Jacobians in log-rate coordinates have eight nonzero numerical singular values. Their condition numbers are '+', '.join(f"{baselines[c]['log_rate_jacobian_condition_number']:.2f}" for c in ['clean','noisy','missing'])+' for clean, noisy, and missing fits. This is a local numerical sensitivity check, not proof of global identifiability or uncertainty calibration.','SmallT')
p('Reproduction record','HeadT')
e=protocol['environment'];cfg=protocol['configuration']
table([['Item','Recorded setting'],
    ['Network','Three hidden tanh layers; width 48; eight logits; one global softmax'],
    ['State transform','Blend exact initial state with simplex output; gate 1 - exp(-t/3.2)'],
    ['Loss','Observed-state MSE + 0.0001 x normalized-time equation MSE'],
    ['Optimizer','8,000 Adam updates at learning rate 0.001; up to 1,000 L-BFGS iterations'],
    ['Grids / integration','161 training collocation points; 641 diagnostic points; DOP853 rtol 1e-10, atol 1e-12'],
    ['Environment',f"Python {e['python']}; torch {e['torch']}; NumPy {e['numpy']}; SciPy {e['scipy']}; float64; CPU; one thread per process"],
    ['Recorded verification','Source and data hashes; checkpoint diagnostics; objective recomputation; model identities; limiting cases; independent solvers']],[116,352])
p('Source: src/two_region_sird.py. Experiment runner: scripts/validate_step3.py. Model checks: scripts/check_step3_model.py. The companion notebook displays the recorded results and can retrain an illustrative clean seed. protocol.json preserves source hashes, data hashes, starting values, and the fixed design; the results archive retains every final run. Fixed training budgets do not certify optimizer convergence.','SmallT')
p('Remaining scope','HeadT')
p('No real mobility or surveillance dataset, compartment-specific movement, unknown initial-state fit, uncertainty interval, or real-data forecast is included. The multipatch SIS extension remains planned. Step 4 is repository documentation and GitHub publication; rebuilding Exhibit 12.3 and reconciling Draft31 remain subsequent stages.','SmallT')

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d5dce2'));canvas.line(54,42,558,42)
    canvas.setFont('TSans',7.3);canvas.drawString(54,29,'Kingsley Chimuanya Ukandu | DINN Step 3 Mobility Validation | October 11, 2026')
    canvas.drawRightString(558,29,f'Page {doc.page}');canvas.restoreState()
pdf=out/'DINN_Step3_Two_Region_Mobility_Validation.pdf'
SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=72,rightMargin=72,topMargin=50,bottomMargin=54,
    title='Two-Region SIRD Mobility Validation',author='Kingsley Chimuanya Ukandu').build(story,onFirstPage=footer,onLaterPages=footer)
archive=Path('output/DINN_Step3_Mobility_Results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for file in sorted(DATA.rglob('*')):
        if file.is_file() and file.suffix in ['.json','.csv','.png']:
            z.write(file,file.relative_to(DATA))
    z.writestr('READ_THIS.txt','Step 3 two-region SIRD mobility validation. Read protocol.json and model_checks.json first.\n'
        'All nine final neural runs and three ODE baselines are included. The development pilot is identified separately.\n'
        'States are fractions of combined initial population. Deceased states do not move. Transmission uses fixed initial regional reference populations.\n'
        'No real-data performance or method superiority is claimed. Repository source is prepared separately for the planned GitHub publication stage.\n')
print(json.dumps({'report':str(pdf),'results_archive':str(archive)}))
