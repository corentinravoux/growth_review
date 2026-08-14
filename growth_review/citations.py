"""BibTeX keys for every compiled measurement, and caption builders.

The keys below are those of ``bibli_hdr_0.bib`` (the HDR manuscript's
bibliography). Every one was established by matching the arXiv identifier in the
data file against the arXiv number inside each bib entry and then confirming by
title -- none was inferred from the shape of the key. Two that a name-based
guess would have got wrong:

  * ``qin_redshift-space_2025`` is Momentum Power Spectrum *III*, arXiv:2411.09571,
    i.e. Qin2024 here -- not the 2019 paper.
  * ``boubel_large-scale_2024`` (arXiv:2301.12648) is a different paper from
    ``boubel_cosmic_2023``.

If you use this package against another bibliography, replace the two dicts and
nothing else: the caption builders read them and refuse to emit an uncited
point.
"""
from .methods import FAMILY_LABEL, FAMILY_ORDER

# Keyed by the `key` column of the PV compilation.
PV_BIBKEY = {
    "Johnson2014": "johnson_6df_2014",
    "Howlett2017": "howlett_2mtf_2017",
    "AdamsBlake2017": "adams_improving_2017",
    "Carrick2015": "carrick_cosmological_2015",
    "Huterer2017": "huterer_testing_2017",
    "HollingerHudson2024": "hollinger_cosmological_2023",
    "Boruah2020": "boruah_cosmic_2020",
    "Said2020": "said_joint_2020",
    "AdamsBlake2020": "adams_joint_2020",
    "Turner2023": "turner_local_2022",
    "Dupuy2019": "dupuy_estimation_2019",
    "Boubel2024": "boubel_large-scale_2024",
    "Courtois2023_ungrouped": "courtois_gravity_2023",
    "Courtois2023_grouped": "courtois_gravity_2023",
    "Courtois2023_snia": "courtois_gravity_2023",
    "Lai2023": "lai_using_2022",
    "Qin2024": "qin_redshift-space_2025",
    "Stahl2021": "stahl_peculiar-velocity_2021",
    "DESI_DR1_maxlike": "lai_desi_2026",
    "DESI_DR1_powerspec": "qin_desi_2025",
    "DESI_DR1_corrfunc": "turner_desi_2026",
    "DESI_DR1_consensus": "lai_desi_2026,turner_desi_2026,qin_desi_2025",
}

# Compiled rows with no entry in bibli_hdr_0.bib. Not an error -- they are
# simply outside the manuscript's citation set. Listed rather than left implicit
# so `cited_only=True` has an auditable complement.
PV_UNCITED = [
    "LilowNusser2021", "Davis2011", "Nusser2017", "Qin2019", "Franco2026",
    "Lyall2024", "Feix2015", "Feix2017", "Stiskalek2026", "Wang2026_group",
    "Wang2026_local", "MaBranchiniScott2012", "Nguyen2025",
]

# Keyed by the tidy `key` of the RSD compilation, i.e. "<ref>/<label>".
# eBOSS DR16 and DESI DR1 each have two companion papers, both cited.
RSD_BIBKEY = {
    "Percival/2dF": "percival_2df_2004",
    "Blake/WiggleZ": "blake_wigglez_2011",
    "Beutler/6dFGS": "beutler_6df_2012",
    "Howlett/MGS": "howlett_clustering_2015",
    "Okumura/FastSound": "okumura_subaru_2016",
    "Pezzotta/VIPERS": "pezzotta_vimos_2017",
    "eBOSS/DR16_LRG": "bautista_completed_2020,gil-marin_completed_2020",
    "eBOSS/DR16_ELG": "tamone_completed_2020,mattia_completed_2020",
    "eBOSS/DR16_QSO": "hou_completed_2020,neveux_completed_2020",
    "DESI-Y1/BGS": "adame_desi_2025,collaboration_desi_2025-2",
    "DESI-Y1/LRG1": "adame_desi_2025,collaboration_desi_2025-2",
    "DESI-Y1/LRG2": "adame_desi_2025,collaboration_desi_2025-2",
    "DESI-Y1/LRG3": "adame_desi_2025,collaboration_desi_2025-2",
    "DESI-Y1/ELG2": "adame_desi_2025,collaboration_desi_2025-2",
    "DESI-Y1/QSO": "adame_desi_2025,collaboration_desi_2025-2",
    # Deliberately unmapped rather than guessed: Alam BOSS DR12, Zarrouk DR14
    # QSO, Icaza-Lizaola DR14 LRG have no entry in bibli_hdr_0.bib.
}

BIBKEY = {**PV_BIBKEY, **RSD_BIBKEY}


def cited_only(df):
    """Restrict a tidy frame to rows that have a bibliography entry."""
    return df[df["key"].isin(BIBKEY)].copy()


def uncited(df):
    """The complement: rows that would be plotted but could not be cited."""
    return sorted(set(df["key"]) - set(BIBKEY))


def _cite(keys):
    return r"\cite{" + keys + "}" if keys else r"\textbf{[NO CITATION]}"


def _dedup(keys):
    return ",".join(dict.fromkeys(k for k in keys if k))


def cite_rows(df):
    """A single \\cite{...} for every row of `df`, plus the uncitable keys."""
    keys, missing = [], []
    for k in df["key"]:
        bk = BIBKEY.get(k)
        if bk is None:
            missing.append(k)
        else:
            keys.extend(bk.split(","))
    return _cite(_dedup(keys)), sorted(set(missing))


def caption_pv(df):
    """Caption body for a PV figure grouped by method family.

    Returns (body, consensus_cite, missing). `body` is a semicolon-separated
    list of "<family> \\cite{...}" clauses; the consensus row is cited
    separately because it is plotted as its own series and would otherwise be
    drawn but never cited. `missing` is non-empty if any plotted point has no
    bibliography entry -- treat that as a failure, not a warning.
    """
    parts, missing, consensus = [], [], ""
    for fam in FAMILY_ORDER:
        sub = df[df["method_family"] == fam]
        if not len(sub):
            continue
        cite, miss = cite_rows(sub)
        missing.extend(miss)
        if fam == "consensus":
            consensus = cite
        else:
            parts.append(f"{FAMILY_LABEL[fam].lower()}~{cite}")
    other = df[~df["method_family"].isin(FAMILY_ORDER)]
    if len(other):
        cite, miss = cite_rows(other)
        missing.extend(miss)
        parts.append(f"other~{cite}")
    return "; ".join(parts), consensus, sorted(set(missing))


def caption_rsd(df):
    """Caption citation list for an RSD figure, in redshift order."""
    return cite_rows(df.sort_values("z"))
