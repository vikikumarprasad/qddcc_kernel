"""Build Table S2 from complete balanced cc-pVDZ runs, without score cutoffs.

Usage: python Experiments/make_table_s2.py
"""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'Experiments/results/water_learning_curve'
ORDER=['Classical','Chebyshev','MultiControl','Hubregtsen']


def main():
    summary=pd.read_csv(SOURCE/'summary.csv')
    assert len(summary)==28 and (summary.repetitions==5).all()
    reference=pd.read_csv(SOURCE/'energy_references.csv')
    mp2=1000*(reference.mp2_archive-reference.ccsd_reference).abs().mean()
    lines=[r'\begin{table}[H]',
        r'\caption{Test $R^2$ and mean absolute deviation from the CCSD correlation energy for the cc-pVDZ water benchmark (Figure 4b). Entries are means $\pm$ standard errors over five repetitions of training-sample selection and, for PQKs, measurement sampling; circuit parameters are held fixed. Every run predicts all 231,040 amplitudes of the 40 held-out records. No run is excluded on the basis of its score. The MP2 deviation is '+f'{mp2:.2f}'+r' m$E_h$.}',
        r'\label{tab:ccpvdz}',r'\centering',r'\small',r'\begin{tabular}{rllll}',r'\toprule',
        r'$N_\mathrm{train}$ & Classical (RBF) & ChebyshevPQC & MultiControl & Hubregtsen \\',r'\midrule']
    for metric,title,decimals in [('r2',r'Test $R^2$',3),('energy',r'Deviation from $E_\mathrm{CCSD}^\mathrm{corr}$ (m$E_h$)',2)]:
        lines.append(r'\multicolumn{5}{l}{'+title+r'} \\')
        for n,g in summary.groupby('n_train'):
            g=g.set_index('family'); cells=[]
            for f in ORDER:
                mean=g.loc[f, 'r2_mean' if metric=='r2' else 'energy_mean_mEh']
                sem=g.loc[f, 'r2_sem' if metric=='r2' else 'energy_sem_mEh']
                cells.append(f'${mean:.{decimals}f}'+r'\pm'+f'{sem:.{decimals}f}$')
            lines.append(f'{n:,} & '+' & '.join(cells)+r' \\')
        lines.append(r'\midrule' if metric=='r2' else r'\bottomrule')
    lines += [r'\end{tabular}',r'\end{table}']
    out=ROOT/'paper_results/regenerated_tables/table_s2.tex';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines)+'\n')
    print(out)


if __name__=='__main__':main()
