"""Hand-curated reference data for name perturbation and blocking (plan sections 2.1, 3).

These three tables are the domain knowledge in this project. A generic
implementation would omit them and its recall on transliteration and
legal-suffix variants would be poor. They belong in version control where a
reviewer can see them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Transliteration equivalence groups.
#
# Each frozenset is a set of spellings that refer to the same underlying name.
# Romanisation of Arabic, Persian, Cyrillic and Chinese names is not
# standardised, so the same person is spelled differently by different banks,
# news outlets and government agencies. This is the single largest source of
# real name variance in sanctions screening.
#
# Used in two directions:
#   - perturbation: replace a token with a sibling from its group
#   - blocking (optional): expand a query token to its whole group
# ---------------------------------------------------------------------------
TRANSLITERATION_GROUPS: tuple[frozenset[str], ...] = (
    # Arabic / Persian given names
    frozenset({"MOHAMMED", "MOHAMED", "MUHAMMAD", "MUHAMED", "MOHAMMAD", "MEHMET", "MOHD"}),
    frozenset({"AHMED", "AHMAD", "AHMET", "AHMOUD"}),
    frozenset({"ABDUL", "ABDEL", "ABD AL", "ABDALLA", "ABDULLAH", "ABDALLAH", "ABDULLA"}),
    frozenset({"HUSSEIN", "HUSSAIN", "HUSAYN", "HUSEIN", "HOSSEIN"}),
    frozenset({"HASSAN", "HASAN", "HASSEN"}),
    frozenset({"YUSUF", "YOUSEF", "YOUSSEF", "YUSSEF", "JOSEPH"}),
    frozenset({"IBRAHIM", "IBRAHEEM", "EBRAHIM"}),
    frozenset({"KHALID", "KHALED", "HALID"}),
    frozenset({"OMAR", "UMAR", "OMER"}),
    frozenset({"ALI", "ALY", "ALEE"}),
    frozenset({"SAEED", "SAID", "SAYYID", "SAYED", "SAEID"}),
    frozenset({"JAMAL", "GAMAL", "DJAMAL"}),
    frozenset({"FAISAL", "FAYSAL", "FEISAL"}),
    frozenset({"MAHMOUD", "MAHMUD", "MAHMOOD"}),
    frozenset({"RASHID", "RASHEED", "RACHID"}),
    frozenset({"SULEIMAN", "SULAIMAN", "SOLEIMANI", "SULAYMAN"}),
    frozenset({"TARIQ", "TAREQ", "TARIK", "TAREK"}),
    frozenset({"ZAYN", "ZAIN", "ZEIN", "ZAYNE"}),
    # Arabic particles (also handled as droppable tokens)
    frozenset({"BIN", "IBN", "BEN"}),
    frozenset({"AL", "EL", "AL-", "EL-"}),
    # Cyrillic-origin
    frozenset({"ALEKSANDR", "ALEXANDER", "ALEXANDR", "OLEKSANDR"}),
    frozenset({"DMITRY", "DMITRI", "DMITRII", "DIMITRY"}),
    frozenset({"SERGEY", "SERGEI", "SERGUEI", "SERHIY"}),
    frozenset({"YEVGENY", "EVGENY", "EVGENII", "YEVGENIY"}),
    frozenset({"MIKHAIL", "MICHAIL", "MYKHAILO"}),
    frozenset({"VIKTOR", "VICTOR"}),
    frozenset({"IGOR", "IHOR"}),
    frozenset({"OOO", "LLC", "O.O.O."}),          # Russian limited-liability form
    frozenset({"ZAO", "OAO", "PAO", "JSC", "OJSC"}),
    frozenset({"STROY", "STROI"}),
    frozenset({"NEFT", "NEFTE", "NAFT"}),
    # Chinese romanisation
    frozenset({"ZHANG", "CHANG"}),
    frozenset({"LI", "LEE", "LY"}),
    frozenset({"WANG", "WONG"}),
    frozenset({"CHEN", "CHAN"}),
    frozenset({"HUANG", "HWANG"}),
)

# token -> sorted list of sibling spellings
TRANSLITERATION_MAP: dict[str, tuple[str, ...]] = {}
for _group in TRANSLITERATION_GROUPS:
    for _tok in _group:
        TRANSLITERATION_MAP[_tok] = tuple(sorted(_group - {_tok}))

# Name particles that customer records frequently omit.
DROPPABLE_PARTICLES = frozenset({"BIN", "IBN", "BEN", "AL", "EL", "ABU", "BINT", "VAN", "VON", "DE", "DA", "DEL"})


# ---------------------------------------------------------------------------
# 2. Legal-form suffixes.
#
# Canonicalised away before entity name comparison, but kept as a separate
# field: a suffix implies a jurisdiction (OOO -> Russia, FZE -> UAE), which is
# weak corroborating evidence the adjudicator may use. Never discard it.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES: tuple[str, ...] = (
    "LLC", "L.L.C.", "L L C", "LTD", "LTD.", "LIMITED", "CO", "CO.", "COMPANY",
    "INC", "INC.", "INCORPORATED", "CORP", "CORP.", "CORPORATION",
    "PLC", "GMBH", "MBH", "AG", "SA", "S.A.", "SAS", "SARL", "S.A.R.L.",
    "BV", "B.V.", "NV", "N.V.", "AB", "AS", "A/S", "OY", "SPA", "S.P.A.", "SRL",
    "PJSC", "JSC", "OJSC", "CJSC", "OOO", "ZAO", "OAO", "PAO",
    "FZE", "FZCO", "FZ-LLC", "DMCC", "LLP", "LP", "PTE", "PTE.", "PVT", "PVT.",
    "SDN BHD", "BHD", "TRUST", "FOUNDATION", "GROUP HOLDING",
)

# Suffixes safe to substitute for one another when perturbing (same rough role).
SUBSTITUTABLE_SUFFIXES: tuple[str, ...] = (
    "LLC", "LTD", "LIMITED", "INC", "CORP", "CO", "GMBH", "SA", "BV",
    "PJSC", "JSC", "OOO", "FZE", "FZCO", "DMCC", "PTE", "PLC",
)

# Suffix -> jurisdiction it implies. Weak evidence only.
SUFFIX_JURISDICTION_HINT: dict[str, str] = {
    "OOO": "Russia", "ZAO": "Russia", "OAO": "Russia", "PAO": "Russia",
    "GMBH": "Germany", "AG": "Germany/Switzerland", "MBH": "Austria",
    "FZE": "United Arab Emirates", "FZCO": "United Arab Emirates",
    "DMCC": "United Arab Emirates", "PJSC": "United Arab Emirates/Russia",
    "SARL": "France/Luxembourg", "SAS": "France", "BV": "Netherlands",
    "NV": "Netherlands/Belgium", "PTE": "Singapore", "SDN BHD": "Malaysia",
    "SPA": "Italy", "SRL": "Italy", "OY": "Finland", "AB": "Sweden",
}


# ---------------------------------------------------------------------------
# 3. Generic corporate tokens.
#
# These carry no identity. Two unrelated firms both called "... General
# Trading ..." are not related. Kept as a hand-written floor; the real weights
# come from the IDF table computed over the loaded entity names (build_idf in
# make_benchmark.py), because "PETROLEUM" is generic on a list dominated by
# energy sanctions but distinctive elsewhere. Use both: IDF for scoring, this
# list for the prompt rule and for choosing acronym letters.
# ---------------------------------------------------------------------------
GENERIC_CORPORATE_TOKENS: frozenset[str] = frozenset({
    "GENERAL", "TRADING", "TRADE", "INTERNATIONAL", "INTL", "GROUP", "HOLDING",
    "HOLDINGS", "COMPANY", "ENTERPRISE", "ENTERPRISES", "INDUSTRIES",
    "INDUSTRIAL", "SERVICES", "SERVICE", "SUPPLY", "IMPORT", "EXPORT",
    "COMMERCIAL", "COMMERCE", "BUSINESS", "DEVELOPMENT", "INVESTMENT",
    "INVESTMENTS", "MANAGEMENT", "PARTNERS", "ASSOCIATES", "SOLUTIONS",
    "SYSTEMS", "TECHNOLOGY", "TECHNOLOGIES", "GLOBAL", "WORLD", "WORLDWIDE",
    "NATIONAL", "STATE", "PUBLIC", "PRIVATE", "JOINT", "STOCK", "AND", "OF",
    "THE", "FOR", "CENTER", "CENTRE", "OFFICE", "AGENCY", "BUREAU", "FIRM",
})
