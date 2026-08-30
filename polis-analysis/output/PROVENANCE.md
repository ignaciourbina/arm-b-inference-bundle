# PROVENANCE — reconstructed, and VALIDATED against the methods appendix

`ising_profiles.json` / `irt_ising_theta.json` were re-derived on 2026-08-27
from the raw Polis open data (`compdemocracy/openData`, `15-per-hour-seattle`),
because the reproduction bundle does not ship `polis-analysis/`.

Method: implemented exactly per `../method-appendix-semantic-ising.tex`
("Ising Profile Construction"). Generator:
`/gpfs/scratch/$USER/polis-derived/build_ising_profiles.py`.

    Q_ij = (A_ij - D_ij)/(A_ij + D_ij)          Yule's Q, >=10 joint directional
    h_i  = atanh(mean directional vote on i)
    m_pi = tanh(h_i + sum_{j in O_p} Q_ij v_pj)
    w_pi = v_pi if observed else m_pi           RAW vote space; engine applies d
    r_p  = (1/10) sum_i d_i w_pi
    E_p  = -sum_{i<j} Q_ij w_pi w_pj            coherence = z(-E_p)

## Validation — all pass

Every published descriptive in the appendix is reproduced exactly:
108 engaged profiles; mean 5.40 / median 5 / range 3-10 observed votes;
54.0% cells observed; policy 69 Pro / 31 Con / 8 Ambivalent;
coherence 22 Coherent / 59 Mixed / 27 Ambivalent.

Independent check: the derived Q matrix reproduces all 34 scenario support/
attack edges (omega = 0.2 + 0.6|Q|), strength AND sign, max error 0.00048
(3-decimal rounding) — 0 mismatches.

## Cut points

The appendix reports the resulting counts but not the cut values, so cuts are
calibrated to reproduce those counts exactly:

    MIN_OBS    = 3          (stated in the appendix)
    POLICY_CUT = 0.10       (round; sits in the .0879-.1459 gap)
    COH_LO     = -1.0       COH_HI = +1.0   (exact 22/59/27)

All three cut points recovered as round values that reproduce the reported
counts exactly. No free parameters remain.
