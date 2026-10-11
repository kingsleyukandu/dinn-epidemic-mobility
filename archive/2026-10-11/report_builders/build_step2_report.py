import json
from pathlib import Path
import zipfile

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path('dinn-project');DATA=ROOT/'validation/step2'
rows=json.loads((DATA/'neural_summary.json').read_text())
protocol=json.loads((DATA/'protocol.json').read_text())
checks=json.loads((DATA/'verification.json').read_text())
forecast=json.loads((DATA/'forecast_summary.json').read_text())
baselines={c:json.loads((DATA/c/'ode_baseline.json').read_text()) for c in ['clean','noisy','missing','forecast']}
out=Path('output/pdf');out.mkdir(parents=True,exist_ok=True)
for name,file in [('VSans','DejaVuSans.ttf'),('VBold','DejaVuSans-Bold.ttf'),('VMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
styles=getSampleStyleSheet()
for name,font,size,leading in [('TitleV','VBold',19,24),('HeadV','VBold',11,15),('BodyV','VSans',9.3,13),('SmallV','VSans',8.2,11),('CellV','VSans',8,10.5),('MonoV','VMono',7.5,10)]:
    styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,spaceAfter=8))
story=[]
def p(s,style='BodyV'):story.append(Paragraph(s,styles[style]))
def table(data,widths):
    t=Table([[Paragraph(str(c),styles['CellV']) for c in r] for r in data],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef2')),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d5dce2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,9)])
def case_rows(case):return [r for r in rows if r['case']==case]
def span(case,key):
    a=[r[key] for r in case_rows(case)];return f'{min(a):.3g} to {max(a):.3g}'
def page(title):story.append(PageBreak());p(title,'TitleV')

p('SIRD DINN Constrained Validation','TitleV')
p('Step 2 | Kingsley Chimuanya Ukandu | October 11, 2026','SmallV')
p('The revised network enforces nonnegative compartments, population conservation, and the exact initial state. All 12 trained runs passed the recorded physical-state checks. Rate recovery remains sensitive to initialization and imperfect observations. Conventional ODE calibration is more accurate in every tested full-window scenario; these results do not demonstrate a DINN advantage.')
p('What changed','HeadV')
table([['Property','Original audited model','Revised implementation'],
    ['States','Unconstrained linear outputs','Softmax simplex output blended with the initial state'],
    ['Initial conditions','Penalty in training loss','Imposed exactly at day zero'],
    ['Population balance','Penalty through equations; measured deviations','Sum of state fractions equals one by construction'],
    ['Equation consistency','Soft residual penalty','Still a soft residual penalty'],
    ['Optimization / precision','20,001 Adam updates; float32','8,000 Adam updates + up to 600 L-BFGS iterations; float64']],[111,162,195])
p('Physical guarantees improve by construction. Precision, optimizer, and reference integration also changed, so comparisons with Step 1 do not isolate the effect of the state constraint on rate accuracy. The original notebook and historical results are preserved.','SmallV')
p('Fixed validation protocol','HeadV')
p('Three network seeds (0, 1, 2), each with the same initial rate guesses: beta 0.8, gamma 0.1, omega 0.15. The true synthetic rates are 0.5, 0.2, and 1/14 per day. All four compartments and the initial state are known. The architecture and optimizer budget were fixed before the validation runs; all final runs are reported, without selecting a favorable seed.')
p('Clean: 54 observations every three days, days 0 to 159. Noisy: the same times with independent Gaussian noise of standard deviation 0.002 in population fraction, fixed random seed 20261011. Measurements were not clipped or renormalized. Missing: 16 interior timepoints removed (29.6%); 38 retained, including both endpoints. Noise and missingness were each tested on one fixed dataset, across the three training seeds.','SmallV')

page('Rate Recovery and ODE Comparison')
p('Absolute relative error = 100 x abs(estimate - truth) / abs(truth). RMSE measures population-fraction error against the noiseless reference on 641 times from day 0 to day 160. Values below are rounded; full precision is in the results archive.','SmallV')
table([['Data','Seed','beta error %','gamma error %','omega error %','State RMSE']]+[
    [r['case'].capitalize(),r['seed'],f"{r['beta_error_percent']:.4f}",f"{r['gamma_error_percent']:.4f}",
     f"{r['omega_error_percent']:.4f}",f"{r['trajectory_rmse']:.3g}"]
    for r in rows if r['case']!='forecast'],[64,35,86,91,91,101])
p('Conventional comparator','HeadV')
table([['Data','beta error %','gamma error %','omega error %','State RMSE']]+[
    [c.capitalize(),*[('<1e-9' if baselines[c]['relative_errors_percent'][n]<1e-9 else f"{baselines[c]['relative_errors_percent'][n]:.4f}") for n in ['beta','gamma','omega']],f"{baselines[c]['trajectory_rmse']:.3g}"]
    for c in ['clean','noisy','missing']],[70,100,100,100,98])
p('ODE calibration uses SciPy least squares on log rates with the same observations, initial rate guesses, and known initial state. All four fits converged successfully. Its noiseless fits use the same equations and integration method as synthetic data generation, making this a favorable, model-correct setting for conventional calibration.','SmallV')
story.append(Image(str(DATA/'parameter_errors.png'),width=468,height=168))
p('The dotted line marks 0.1% relative error. The revised clean-data runs exceed it for some rates and seeds. Under the tested noise, errors rise further. No broad noise robustness or estimator-superiority claim follows from these experiments.','SmallV')

page('Physical Constraints and Trajectories')
p('For nonnegative time, let g(t) = 1 - exp(-t/3.2) and q(t) = softmax(network(t/160)). The state is y(t) = (1-g(t)) y0 + g(t) q(t). This is a convex combination of two nonnegative vectors whose entries sum to one. At t = 0, g = 0, so y(0) = y0 exactly. The forecast training network scales its input by 60 rather than 160.','SmallV')
table([['Check across all 12 final runs','Result'],
    ['Negative state entries on diagnostic grids',str(sum(r['negative_entries'] for r in rows))],
    ['Largest population-balance error',f"{max(r['conservation_max_abs_error'] for r in rows):.3g}"],
    ['Largest initial-state error',f"{max(r['initial_max_abs_error'] for r in rows):.3g}"],
    ['Largest per-day equation residual',f"{max(r['residual_max_abs_per_day'] for r in rows):.3g}"],
    ['Saved checkpoint / metric verification','PASS for all 12 runs']],[295,173])
story.append(Image(str(DATA/'clean_trajectories.png'),width=468,height=276))
p('All clean-data seeds are shown against the reference. Positive states and population balance are structural properties; close curve agreement and small residuals are numerical findings. Small decreases in recovered or deceased states occur in 8 of the 12 runs: monotonicity is not enforced. The smallest measured derivative is approximately -0.0000300 population fraction per day. The equation residual is finite, so exact solution of the differential equations is not claimed.','SmallV')

page('Chronological Synthetic Forecast Check')
p('A separate experiment uses only the 21 observations from day 0 through day 60. All residual collocation points also lie within that interval. After fitting, the inferred rates are placed in the SIRD ODE and integrated from the known initial state. Evaluation uses days 61 through 88, a 28-day holdout. These are mechanistic forecasts using inferred rates, not raw neural-network extrapolations.')
table([['Method','Seed','All-state RMSE','Infected MAE']]+[
    ['DINN-inferred rates',r['seed'],f"{r['rmse_all_states']:.6g}",f"{r['infected_mae']:.6g}"] for r in forecast['neural']]+[
    ['ODE calibration','-',f"{forecast['ode']['forecast_rmse_all_states']:.6g}",f"{forecast['ode']['forecast_infected_mae']:.6g}"],
    ['Persistence at day 60','-',f"{forecast['neural'][0]['persistence_rmse_all_states']:.6g}",f"{forecast['neural'][0]['persistence_infected_mae']:.6g}"]],[164,38,133,133])
story.append(Image(str(DATA/'forecast_infected.png'),width=468,height=181))
p('The persistence comparator holds all four day-60 states constant. All error values are population fractions; they are not counts. Forecast rate estimates and their full-precision recovery errors are included in neural_summary.csv. The forecast endpoint is not used for model selection.','SmallV')
p('Scope of this result','HeadV')
p('This is one clean, model-correct synthetic epidemic with a known initial state, one training cutoff, and three optimizer seeds. It checks the training/evaluation time separation and the mechanistic forecasting pipeline. It is not a rolling real-data hindcast, does not measure uncertainty coverage, and does not establish real-world forecasting skill. ODE calibration remains the stronger comparator in this experiment.')

page('Interpretation and Reproduction')
p('Claims supported by this stage','HeadV')
p('The project now has a separate physically constrained SIRD implementation; fixed-budget, multiple-seed tests; one noise condition; one missing-data condition; an ODE calibration comparator; and a synthetic chronological holdout. It supports an honest description of implemented methods and measured limitations. The historical Step 1 parameter-recovery result remains a result of the original implementation, not a uniform guarantee of the revised model.')
p('Remaining technical limits','HeadV')
partial=json.loads((DATA/'partial_observability.json').read_text())
p('All four compartments are observed here. If only susceptible and infected states are observed, recovery and death rates enter their equations through gamma + omega and cannot be separately identified. An alternative split produces S/I differences below '+f"{partial['max_abs_SI_difference']:.3g}"+' in the recorded numerical check, while changing recovered/deceased trajectories. Practical surveillance data therefore require an explicit observation model and an identifiability assessment.')
p('Only three optimizer seeds were tested per condition, and each imperfection uses a single fixed dataset. There are no repeated noise realizations, uncertainty intervals, time-varying rates, unknown initial-state estimates, regional mobility terms, or real surveillance observations in this stage. The synthetic truth generates the data and is used for evaluation only; it does not choose training checkpoints.','SmallV')
p('Recorded configuration and checks','HeadV')
e=protocol['environment']
table([['Item','Recorded setting'],['Network','Three hidden tanh layers, width 32; four softmax logits'],
    ['Loss','Observed-state MSE + 0.0001 x normalized-time equation MSE'],
    ['Collocation','161 points over each training interval'],
    ['ODE reference','DOP853; rtol 1e-10; atol 1e-12'],
    ['Runtime',f"Python {e['python']}; torch {e['torch']}; NumPy {e['numpy']}; SciPy {e['scipy']}; CPU; one thread per process"],
    ['Verification','State-transform bounds, exact initial state, derivative scaling, source/data hashes, all checkpoint metrics, holdout boundaries and baseline objectives']],[110,358])
p('Provenance','HeadV')
p('Original public source commit: 3c6275393c7f2aea437683c0e077996e6bff5d45. The constrained implementation is a new local working version. Its source hash, runner hash, dataset hashes, dependencies, and fixed protocol are recorded in protocol.json. The results archive contains all numerical runs, trajectories, training histories, comparator fits, and verification records.','SmallV')
p('Next planned stage','HeadV')
p('Step 3 is the two-region SIRD mobility extension with conservation and limiting-case checks. Repository publication and commits, the formal exhibit, and petition wording reconciliation remain later stages. This report is a development and validation record; it does not itself revise Draft31.','SmallV')

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d5dce2'));canvas.line(54,42,558,42)
    canvas.setFont('VSans',7.3);canvas.drawString(54,29,'Kingsley Chimuanya Ukandu | DINN Step 2 Validation | October 11, 2026')
    canvas.drawRightString(558,29,f'Page {doc.page}');canvas.restoreState()
pdf=out/'DINN_Step2_Constrained_Validation.pdf'
SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=72,rightMargin=72,topMargin=50,bottomMargin=54,
    title='SIRD DINN Constrained Validation',author='Kingsley Chimuanya Ukandu').build(story,onFirstPage=footer,onLaterPages=footer)

# Preserve numerical evidence without duplicating repository-backed source code.
zip_path=Path('output/DINN_Step2_Validation_Results.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(DATA.rglob('*')):
        if path.is_file() and path.suffix in ['.csv','.json','.png']:
            z.write(path,path.relative_to(DATA))
    z.writestr('READ_THIS.txt','Step 2 controlled synthetic validation. Read protocol.json first. All seeds retained.\n'
        'The archive contains numerical evidence and plots; repository code is prepared separately for the planned GitHub publication stage.\n'
        'No real-world validation or estimator superiority is claimed.\n')
print(json.dumps({'report':str(pdf),'results':str(zip_path)}))
