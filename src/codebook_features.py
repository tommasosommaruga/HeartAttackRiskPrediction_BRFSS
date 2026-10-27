"""Recode BRFSS 2015 columns before mode fill and before any column score.

Feature index i is the column of the array from `helpers.load_csv_data`.
That loader removes `Id`. `FMONTH` is 1, `IYEAR` is 5, and `WEIGHT2` is 62.

Do-not-know and refused codes become missing here. The mode fill in
`src.preprocessing` runs after this function and must not see 7, 9, 77, 99,
7777, or 9999 as if they were real magnitudes. `BPMEDS` is the exception
that is filled by a rule instead of the mode: a person whose `BPHIGH4` is
2, 3, or 4 is not a treated hypertensive, so the medication item is 0.
"""

import textwrap

import numpy as np


HEADER = (
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
    "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2",
    "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "GENHLTH", "PHYSHLTH", "MENTHLTH",
    "POORHLTH", "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4", "BPMEDS",
    "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3", "ASTHNOW", "CHCSCNCR",
    "CHCOCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY", "DIABETE3",
    "DIABAGE2", "SEX", "MARITAL", "EDUCA", "RENTHOM1", "NUMHHOL2", "NUMPHON2",
    "CPDEMO1", "VETERAN3", "EMPLOY1", "CHILDREN", "INCOME2", "INTERNET", "WEIGHT2",
    "HEIGHT3", "PREGNANT", "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK",
    "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2", "STOPSMK2", "LASTSMK2",
    "USENOW3", "ALCDAY5", "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "FRUITJU1", "FRUIT1",
    "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1", "EXERANY2", "EXRACT11", "EXEROFT1",
    "EXERHMM1", "EXRACT21", "EXEROFT2", "EXERHMM2", "STRENGTH", "LMTJOIN3",
    "ARTHDIS2", "ARTHSOCL", "JOINPAIN", "SEATBELT", "FLUSHOT6", "FLSHTMY2",
    "IMFVPLAC", "PNEUVAC3", "HIVTST6", "HIVTSTD3", "WHRTST10", "PDIABTST",
    "PREDIAB1", "INSULIN", "BLDSUGAR", "FEETCHK2", "DOCTDIAB", "CHKHEMO3",
    "FEETCHK", "EYEEXAM", "DIABEYE", "DIABEDU", "CAREGIV1", "CRGVREL1", "CRGVLNG1",
    "CRGVHRS1", "CRGVPRB1", "CRGVPERS", "CRGVHOUS", "CRGVMST2", "CRGVEXPT",
    "VIDFCLT2", "VIREDIF3", "VIPRFVS2", "VINOCRE2", "VIEYEXM2", "VIINSUR2",
    "VICTRCT4", "VIGLUMA2", "VIMACDG2", "CIMEMLOS", "CDHOUSE", "CDASSIST", "CDHELP",
    "CDSOCIAL", "CDDISCUS", "WTCHSALT", "LONGWTCH", "DRADVISE", "ASTHMAGE",
    "ASATTACK", "ASERVIST", "ASDRVIST", "ASRCHKUP", "ASACTLIM", "ASYMPTOM",
    "ASNOSLEP", "ASTHMED3", "ASINHALR", "HAREHAB1", "STREHAB1", "CVDASPRN",
    "ASPUNSAF", "RLIVPAIN", "RDUCHART", "RDUCSTRK", "ARTTODAY", "ARTHWGT",
    "ARTHEXER", "ARTHEDU", "TETANUS", "HPVADVC2", "HPVADSHT", "SHINGLE2", "HADMAM",
    "HOWLONG", "HADPAP2", "LASTPAP2", "HPVTEST", "HPLSTTST", "HADHYST2", "PROFEXAM",
    "LENGEXAM", "BLDSTOOL", "LSTBLDS3", "HADSIGM3", "HADSGCO1", "LASTSIG3",
    "PCPSAAD2", "PCPSADI1", "PCPSARE1", "PSATEST1", "PSATIME", "PCPSARS1",
    "PCPSADE1", "PCDMDECN", "SCNTMNY1", "SCNTMEL1", "SCNTPAID", "SCNTWRK1",
    "SCNTLPAD", "SCNTLWK1", "SXORIENT", "TRNSGNDR", "RCSGENDR", "RCSRLTN2",
    "CASTHDX2", "CASTHNO2", "EMTSUPRT", "LSATISFY", "ADPLEASR", "ADDOWN", "ADSLEEP",
    "ADENERGY", "ADEAT1", "ADFAIL", "ADTHINK", "ADMOVE", "MISTMNT", "ADANXEV",
    "QSTVER", "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_CHISPNC", "_CRACE1", "_CPRACE", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "_LLCPWT",
    "_RFHLTH", "_HCVU651", "_RFHYPE5", "_CHOLCHK", "_RFCHOL", "_LTASTH1", "_CASTHM1",
    "_ASTHMS1", "_DRDXAR1", "_PRACE1", "_MRACE1", "_HISPANC", "_RACE", "_RACEG21",
    "_RACEGR3", "_RACE_G1", "_AGEG5YR", "_AGE65YR", "_AGE80", "_AGE_G", "HTIN4",
    "HTM4", "WTKG3", "_BMI5", "_BMI5CAT", "_RFBMI5", "_CHLDCNT", "_EDUCAG",
    "_INCOMG", "_SMOKER3", "_RFSMOK3", "DRNKANY5", "DROCDY3_", "_RFBING5",
    "_DRNKWEK", "_RFDRHV5", "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_",
    "ORNGDAY_", "VEGEDA1_", "_MISFRTN", "_MISVEGN", "_FRTRESP", "_VEGRESP",
    "_FRUTSUM", "_VEGESUM", "_FRTLT1", "_VEGLT1", "_FRT16", "_VEG23", "_FRUITEX",
    "_VEGETEX", "_TOTINDA", "METVL11_", "METVL21_", "MAXVO2_", "FC60_", "ACTIN11_",
    "ACTIN21_", "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "_MINAC11",
    "_MINAC21", "STRFREQ_", "PAMISS1_", "PAMIN11_", "PAMIN21_", "PA1MIN_",
    "PAVIG11_", "PAVIG21_", "PA1VIGM_", "_PACAT1", "_PAINDX1", "_PA150R2",
    "_PA300R2", "_PA30021", "_PASTRNG", "_PAREC1", "_PASTAE1", "_LMTACT1",
    "_LMTWRK1", "_LMTSCL1", "_RFSEAT2", "_RFSEAT3", "_FLSHOT6", "_PNEUMO2",
    "_AIDTST3",
)

INDEX = {name: index for index, name in enumerate(HEADER)}

# Values above this are a handful of extreme body mass rows. The square would
# let them dominate. 60 is already far into the upper tail of this file.
BMI_CAP = 60.0

# These holes are large enough that filling them with the mode hides a refusal.
# The flag is 1 when the recoded value is missing, and it is built before the fill.
# Training also flags every other kept column whose missing rate reaches
# `missing_flag_min_rate` in config/features.json. The names below are the
# floor: they are flagged even when that rate is set higher.
MISSING_FLAGS = (
    ("_INCOMG", "incomg_missing"),
    ("_RFCHOL", "rfchol_missing"),
    ("_FRTLT1", "frtlt1_missing"),
    ("_VEGLT1", "veglt1_missing"),
    ("_RFSEAT3", "rfseat3_missing"),
    ("_PASTRNG", "pastrng_missing"),
    ("_AIDTST3", "aidtst3_missing"),
)

# Ordered maps. Any code that is not listed, including 7, 9, 77, and 99, is missing.
_ORDINAL = {
    "GENHLTH": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "HLTHPLN1": {1: 1, 2: 0},
    "PERSDOC2": {1: 1, 2: 1, 3: 0},
    "MEDCOST": {1: 1, 2: 0},
    "CHECKUP1": {1: 1, 2: 2, 3: 3, 4: 4, 8: 5},
    "CVDSTRK3": {1: 1, 2: 0},
    "CHCSCNCR": {1: 1, 2: 0},
    "CHCOCNCR": {1: 1, 2: 0},
    "CHCCOPD1": {1: 1, 2: 0},
    "ADDEPEV2": {1: 1, 2: 0},
    "CHCKIDNY": {1: 1, 2: 0},
    "SEX": {1: 1, 2: 0},
    "VETERAN3": {1: 1, 2: 0},
    "QLACTLM2": {1: 1, 2: 0},
    "USEEQUIP": {1: 1, 2: 0},
    "BLIND": {1: 1, 2: 0},
    "DECIDE": {1: 1, 2: 0},
    "DIFFWALK": {1: 1, 2: 0},
    "DIFFDRES": {1: 1, 2: 0},
    "DIFFALON": {1: 1, 2: 0},
    "USENOW3": {1: 2, 2: 1, 3: 0},
    "EXERANY2": {1: 1, 2: 0},
    "_RFHYPE5": {2: 1, 1: 0},
    "_CHOLCHK": {1: 1, 2: 0, 3: 0},
    "_RFCHOL": {2: 1, 1: 0},
    "_CASTHM1": {2: 1, 1: 0},
    "_DRDXAR1": {1: 1, 2: 0},
    "_AGE80": None,
    "_BMI5": None,
    "_CHLDCNT": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    "_EDUCAG": {1: 1, 2: 2, 3: 3, 4: 4},
    "_INCOMG": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "_SMOKER3": {1: 3, 2: 2, 3: 1, 4: 0},
    "_RFBING5": {2: 1, 1: 0},
    "_RFDRHV5": {2: 1, 1: 0},
    "_FRTLT1": {1: 1, 2: 0},
    "_VEGLT1": {1: 1, 2: 0},
    "_PASTRNG": {1: 1, 2: 0},
    "_RFSEAT3": {1: 1, 2: 0},
    "_AIDTST3": {1: 1, 2: 0},
    "LMTJOIN3": {1: 1, 2: 0},
    "ARTHDIS2": {1: 1, 2: 0},
    "ARTHSOCL": {1: 1, 2: 2, 3: 3},
    "ARTHWGT": {1: 1, 2: 0},
    "ARTHEXER": {1: 1, 2: 0},
    "CIMEMLOS": {1: 1, 2: 0},
    "PDIABTST": {1: 1, 2: 0},
    "INSULIN": {1: 1, 2: 0},
    "DIABEYE": {1: 1, 2: 0},
    "DIABEDU": {1: 1, 2: 0},
    "EYEEXAM": {1: 1, 2: 2, 3: 3, 4: 4, 8: 5},
    "WTCHSALT": {1: 1, 2: 0},
    "DRADVISE": {1: 1, 2: 0},
    "STOPSMK2": {1: 1, 2: 0},
    "FLUSHOT6": {1: 1, 2: 0},
    "PNEUVAC3": {1: 1, 2: 0},
    "_FLSHOT6": {1: 1, 2: 0},
    "_PNEUMO2": {1: 1, 2: 0},
    "SHINGLE2": {1: 1, 2: 0},
    "BLDSTOOL": {1: 1, 2: 0},
    "HADSIGM3": {1: 1, 2: 0},
    "ADANXEV": {1: 1, 2: 0},
    "CAREGIV1": {1: 1, 2: 0},
    "EMTSUPRT": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "LSATISFY": {1: 1, 2: 2, 3: 3, 4: 4},
    "SCNTMNY1": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "SCNTMEL1": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
}

# low, high, and codes that mean zero. Anything else, including 77 and 99, is missing.
_BOUNDED = {
    "POORHLTH": (1, 30, (88,)),
    "JOINPAIN": (0, 10, ()),
    "DIABAGE2": (1, 97, ()),
    "CHKHEMO3": (1, 76, (88,)),
    "DOCTDIAB": (1, 76, (88,)),
    "AVEDRNK2": (1, 76, ()),
    "DRNK3GE5": (1, 76, (88,)),
    "MAXDRNKS": (1, 76, ()),
    "ADPLEASR": (1, 14, (88,)),
    "ADDOWN": (1, 14, (88,)),
    "ADSLEEP": (1, 14, (88,)),
    "ADENERGY": (1, 14, (88,)),
    "ADEAT1": (1, 14, (88,)),
    "ADFAIL": (1, 14, (88,)),
    "ADTHINK": (1, 14, (88,)),
    "ADMOVE": (1, 14, (88,)),
}

_INDICATORS = {
    "DIABETE3": (
        (1, "diab_yes"),
        (2, "diab_pregnancy"),
        (4, "diab_pre"),
    ),
    "MARITAL": (
        (2, "marital_divorced"),
        (3, "marital_widowed"),
        (4, "marital_separated"),
        (5, "marital_never"),
        (6, "marital_couple"),
    ),
    "RENTHOM1": (
        (2, "housing_rent"),
        (3, "housing_other"),
    ),
    "EMPLOY1": (
        (2, "employ_self"),
        (3, "employ_out_long"),
        (4, "employ_out_short"),
        (5, "employ_homemaker"),
        (6, "employ_student"),
        (7, "employ_retired"),
        (8, "employ_unable"),
    ),
    "_RACEGR3": (
        (2, "race_black"),
        (3, "race_other"),
        (4, "race_multi"),
        (5, "race_hispanic"),
    ),
}

_INDICATOR_VALID = {
    "DIABETE3": (1, 2, 3, 4),
    "MARITAL": (1, 2, 3, 4, 5, 6),
    "RENTHOM1": (1, 2, 3),
    "EMPLOY1": (1, 2, 3, 4, 5, 6, 7, 8),
    "_RACEGR3": (1, 2, 3, 4, 5),
}

# Output order. Multi-column items contribute every indicator, reference last as zeros.
PLAN = (
    ("GENHLTH", "map"),
    ("PHYSHLTH", "days"),
    ("MENTHLTH", "days"),
    ("HLTHPLN1", "map"),
    ("PERSDOC2", "map"),
    ("MEDCOST", "map"),
    ("CHECKUP1", "map"),
    ("BPMEDS", "bpmeds"),
    ("_RFHYPE5", "map"),
    ("_CHOLCHK", "map"),
    ("_RFCHOL", "map"),
    ("CVDSTRK3", "map"),
    ("_CASTHM1", "map"),
    ("CHCSCNCR", "map"),
    ("CHCOCNCR", "map"),
    ("CHCCOPD1", "map"),
    ("_DRDXAR1", "map"),
    ("ADDEPEV2", "map"),
    ("CHCKIDNY", "map"),
    ("DIABETE3", "indicators"),
    ("QLACTLM2", "map"),
    ("USEEQUIP", "map"),
    ("BLIND", "map"),
    ("DECIDE", "map"),
    ("DIFFWALK", "map"),
    ("DIFFDRES", "map"),
    ("DIFFALON", "map"),
    ("_SMOKER3", "map"),
    ("USENOW3", "map"),
    ("EXERANY2", "map"),
    ("_PASTRNG", "map"),
    ("_RFBING5", "map"),
    ("_RFDRHV5", "map"),
    ("_FRTLT1", "map"),
    ("_VEGLT1", "map"),
    ("_RFSEAT3", "map"),
    ("_AIDTST3", "map"),
    ("SEX", "map"),
    ("_AGE80", "age"),
    ("_BMI5", "bmi"),
    ("_EDUCAG", "map"),
    ("_INCOMG", "map"),
    ("_CHLDCNT", "map"),
    ("VETERAN3", "map"),
    ("EMPLOY1", "indicators"),
    ("MARITAL", "indicators"),
    ("RENTHOM1", "indicators"),
    ("_RACEGR3", "indicators"),
    ("POORHLTH", "bounded"),
    ("LMTJOIN3", "map"),
    ("ARTHDIS2", "map"),
    ("ARTHSOCL", "map"),
    ("JOINPAIN", "bounded"),
    ("ARTHWGT", "map"),
    ("ARTHEXER", "map"),
    ("CIMEMLOS", "map"),
    ("PDIABTST", "map"),
    ("INSULIN", "map"),
    ("DIABEYE", "map"),
    ("DIABEDU", "map"),
    ("DIABAGE2", "bounded"),
    ("CHKHEMO3", "bounded"),
    ("DOCTDIAB", "bounded"),
    ("EYEEXAM", "map"),
    ("WTCHSALT", "map"),
    ("DRADVISE", "map"),
    ("AVEDRNK2", "bounded"),
    ("DRNK3GE5", "bounded"),
    ("MAXDRNKS", "bounded"),
    ("STOPSMK2", "map"),
    ("FLUSHOT6", "map"),
    ("PNEUVAC3", "map"),
    ("_FLSHOT6", "map"),
    ("_PNEUMO2", "map"),
    ("SHINGLE2", "map"),
    ("BLDSTOOL", "map"),
    ("HADSIGM3", "map"),
    ("ADANXEV", "map"),
    ("CAREGIV1", "map"),
    ("PREGNANT", "pregnant"),
    ("EMTSUPRT", "map"),
    ("LSATISFY", "map"),
    ("SCNTMNY1", "map"),
    ("SCNTMEL1", "map"),
    ("ADPLEASR", "bounded"),
    ("ADDOWN", "bounded"),
    ("ADSLEEP", "bounded"),
    ("ADENERGY", "bounded"),
    ("ADEAT1", "bounded"),
    ("ADFAIL", "bounded"),
    ("ADTHINK", "bounded"),
    ("ADMOVE", "bounded"),
)

# Kept even though most rows were not asked. A missing indicator is added
# before the mode fill, so the blank is a column and not an invented answer.
_SPARSE_KEEP = frozenset((
    "POORHLTH", "LMTJOIN3", "ARTHDIS2", "ARTHSOCL", "JOINPAIN", "ARTHWGT",
    "ARTHEXER", "CIMEMLOS", "PDIABTST", "INSULIN", "DIABEYE", "DIABEDU",
    "DIABAGE2", "CHKHEMO3", "DOCTDIAB", "EYEEXAM", "WTCHSALT", "DRADVISE",
    "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "STOPSMK2", "FLUSHOT6", "PNEUVAC3",
    "_FLSHOT6", "_PNEUMO2", "SHINGLE2", "BLDSTOOL", "HADSIGM3", "ADANXEV",
    "CAREGIV1", "PREGNANT", "EMTSUPRT", "LSATISFY", "SCNTMNY1", "SCNTMEL1",
    "ADPLEASR", "ADDOWN", "ADSLEEP", "ADENERGY", "ADEAT1", "ADFAIL",
    "ADTHINK", "ADMOVE",
))


def _survey(detail):
    return (
        detail
        + " Survey machinery, not a health measure. Removed before the column "
        + "score, so the score cannot put it back."
    )


def _copy(detail, kept):
    return (
        detail
        + " The same fact is kept as "
        + kept
        + ", so this copy is not also a column."
    )


def _skip(detail):
    return (
        detail
        + " A blank means the question was not asked, not a measured zero. "
        + "Mode fill would invent an answer, so the column is removed first."
    )


def _build_reasons():
    """One reason for every CSV column after Id. Keep text includes the recode."""
    reasons = {}

    def put(name, action, text):
        if name not in INDEX:
            raise RuntimeError("unknown column " + name)
        if name in reasons:
            raise RuntimeError("repeated column " + name)
        reasons[name] = (action, text)

    put("_STATE", "drop", _survey(
        "State FIPS code. It is the sampling geography. As one number the "
        "codes are not an ordered exposure."
    ))
    put("FMONTH", "drop", _survey(
        "Interview month, 1 through 12. It is the survey calendar."
    ))
    put("IDATE", "drop", _survey(
        "Interview date packed as a month-day-year integer."
    ))
    put("IMONTH", "drop", _survey("Interview month, a second copy of the calendar."))
    put("IDAY", "drop", _survey("Interview day of the month."))
    put("IYEAR", "drop", _survey(
        "Interview year. Almost every row is 2015, and the rest are 2016 "
        "fieldwork spillover. It is not age or disease."
    ))
    put("DISPCODE", "drop", _survey(
        "Disposition code, 1100 completed or 1200 partial. It records how the "
        "interview ended."
    ))
    put("SEQNO", "drop", _survey("Sample sequence number. It identifies the interview."))
    put("_PSU", "drop", _survey(
        "Primary sampling unit. In this file it matches the sequence number."
    ))
    put("CTELENUM", "drop", _survey("Landline phone number is correct. A phone check."))
    put("PVTRESD1", "drop", _survey("Private residence flag for the landline sample."))
    put("COLGHOUS", "drop", _survey("College housing flag. Asked only in a tiny skip."))
    put("STATERES", "drop", _survey("Resident of the sampled state. A sample check."))
    put("CELLFON3", "drop", _survey("Has a cell phone. A landline sample question."))
    put("LADULT", "drop", _survey("Adult in the landline household. A sample check."))
    put("NUMADULT", "drop", _survey("Number of adults in the landline household."))
    put("NUMMEN", "drop", _survey("Number of adult men in the landline household."))
    put("NUMWOMEN", "drop", _survey("Number of adult women in the landline household."))
    put("CTELNUM1", "drop", _survey("Cell phone number is correct. A phone check."))
    put("CELLFON2", "drop", _survey("Cell-sample interview flag."))
    put("CADULT", "drop", _survey("Adult on the cell phone. A sample check."))
    put("PVTRESD2", "drop", _survey("Private residence flag for the cell sample."))
    put("CCLGHOUS", "drop", _survey("College housing flag on the cell sample."))
    put("CSTATE", "drop", _survey("Cell respondent lives in the sampled state."))
    put("LANDLINE", "drop", _survey(
        "Has a landline. Codes include 7 and 9, and the item is a sample field."
    ))
    put("HHADULT", "drop", _survey(
        "Adults in the cell-phone household. Code 99 is refused, and the item "
        "is a sample field."
    ))
    put("NUMHHOL2", "drop", _survey("More than one landline in the household."))
    put("NUMPHON2", "drop", _survey(
        "Number of residential phones. Asked on a skip, with 7 and 9 as refusals."
    ))
    put("CPDEMO1", "drop", _survey(
        "Cell-phone ownership question used to sort the landline sample."
    ))
    put("QSTVER", "drop", _survey("Questionnaire version number."))
    put("QSTLANG", "drop", _survey("Questionnaire language."))
    put("MSCODE", "drop", _survey(
        "Metropolitan status code. It is survey geography, not a health measure."
    ))
    put("_STSTR", "drop", _survey(
        "Sample stratum used to build the raking weights."
    ))
    put("_STRWT", "drop", _survey("Stratum weight, a raking input."))
    put("_RAWRAKE", "drop", _survey("Raw raking weight."))
    put("_WT2RAKE", "drop", _survey("Second raking weight."))
    put("_CLLCPWT", "drop", _survey(
        "Combined landline and cell weight used in raking."
    ))
    put("_LLCPWT", "drop", _survey(
        "Final raking weight `_LLCPWT`. It is the survey weight, not body weight."
    ))
    put("_DUALUSE", "drop", _survey(
        "Dual landline and cell use. Code 9 is missing, and the flag adjusts weights."
    ))
    put("_DUALCOR", "drop", _survey(
        "Dual-frame correction factor applied to the sampling weights."
    ))

    put("HAREHAB1", "drop", _skip(
        "Outpatient rehab after a heart attack. That is care given because the "
        "person had the event being predicted, so it leaks the label. "
        "`CVDINFR4` and `CVDCRHD4` are not in this file, and `_MICHD` is not rebuilt."
    ))
    put("STREHAB1", "drop", _skip(
        "Outpatient rehab after a stroke. Removed with the rehab items. It is "
        "treatment after a vascular event. The stroke diagnosis stays as `CVDSTRK3`."
    ))
    put("CVDASPRN", "drop", _skip(
        "Regular aspirin use, from the cardiovascular-health module. Aspirin is "
        "often started after a coronary event, so the answer leaks the label."
    ))
    put("ASPUNSAF", "drop", _skip(
        "Health problem that makes aspirin unsafe. Same module as `CVDASPRN`, "
        "and a blank means the aspirin questions were not asked."
    ))

    put("BPHIGH4", "drop", (
        "Told that blood pressure is high. Codes 1 yes, 2 pregnancy only, "
        "3 no, 4 borderline, 7 do not know, 9 refused. Used only to build "
        "`BPMEDS`: codes 2, 3, and 4 are set to not medicated before the mode "
        "fill. The hypertension fact that stays is `_RFHYPE5`."
    ))
    put("ASTHNOW", "drop", _skip(
        "Still have asthma. Asked only after `ASTHMA3` is yes. Current asthma "
        "is kept as `_CASTHM1`, which already combines the two questions."
    ))
    put("LASTSMK2", "drop", _skip(
        "How long since smoking regularly. Asked of former smokers, and the "
        "codes are not a count of years. `_SMOKER3` keeps former versus never."
    ))
    put("SMOKDAY2", "drop", _skip(
        "Smoke every day, some days, or not at all. Asked only after "
        "`SMOKE100` is yes. `_SMOKER3` already combines those answers, and "
        "code 9 is refused."
    ))
    put("EXRACT11", "drop", _skip(
        "First activity type. Asked only if `EXERANY2` is yes, and the type "
        "codes are not a dose. `EXERANY2` is the kept activity yes or no."
    ))
    put("EXEROFT1", "drop", _skip(
        "How often the first activity. The codes mix times per day and per "
        "week with 777 and 999. `EXERANY2` is kept."
    ))
    put("EXERHMM1", "drop", _skip(
        "How long the first activity, in hours and minutes packed into one "
        "integer, with 777 and 999 as missing. `EXERANY2` is kept."
    ))
    put("EXRACT21", "drop", _skip(
        "Second activity type. Blank often means there was no second activity. "
        "`EXERANY2` is kept."
    ))
    put("EXEROFT2", "drop", _skip(
        "How often the second activity, with the same packed frequency codes. "
        "`EXERANY2` is kept."
    ))
    put("EXERHMM2", "drop", _skip(
        "How long the second activity, packed hours and minutes. `EXERANY2` is kept."
    ))

    for name, detail in (
        ("LMTJOIN3", "Limited by joint symptoms."),
        ("ARTHDIS2", "Arthritis limits usual activities."),
        ("ARTHSOCL", "Arthritis limits social activities."),
        ("JOINPAIN", "Joint pain on a 0 to 10 scale, with 99 refused."),
        ("FLUSHOT6", "Flu shot in the past year."),
        ("FLSHTMY2", "Month and year of the flu shot, packed into one integer."),
        ("IMFVPLAC", "Place the flu shot was given."),
        ("PNEUVAC3", "Pneumonia vaccination."),
        ("HIVTSTD3", "Month and year of the last HIV test."),
        ("WHRTST10", "Where the last HIV test was done."),
        ("PDIABTST", "Tested for high blood sugar in the past three years."),
        ("PREDIAB1", "Told prediabetes in the diabetes module. `diab_pre` is kept."),
        ("INSULIN", "Now taking insulin. Asked of people with diabetes."),
        ("BLDSUGAR", "How often blood sugar is checked."),
        ("FEETCHK2", "How often feet are checked for sores."),
        ("DOCTDIAB", "Times a professional was seen for diabetes."),
        ("CHKHEMO3", "Times A1C was checked."),
        ("FEETCHK", "Times feet were checked by a professional."),
        ("EYEEXAM", "Last eye exam dilated, asked of people with diabetes."),
        ("DIABEYE", "Told diabetes affected the eyes."),
        ("DIABEDU", "Taken a class on managing diabetes."),
        ("CAREGIV1", "Provided care to a friend or family member."),
        ("CRGVREL1", "Relationship to the person cared for."),
        ("CRGVLNG1", "How long care has been given."),
        ("CRGVHRS1", "Hours of care per week."),
        ("CRGVPRB1", "Main health problem of the person cared for."),
        ("CRGVPERS", "Managed personal care."),
        ("CRGVHOUS", "Helped with household tasks."),
        ("CRGVMST2", "Greatest difficulty as a caregiver."),
        ("CRGVEXPT", "Expected to keep providing care."),
        ("VIDFCLT2", "Difficulty seeing, even with glasses."),
        ("VIREDIF3", "Last time vision was checked."),
        ("VIPRFVS2", "Visited an eye professional."),
        ("VINOCRE2", "Told to get an eye exam."),
        ("VIEYEXM2", "Last dilated eye exam."),
        ("VIINSUR2", "Eye-care insurance."),
        ("VICTRCT4", "Cataract."),
        ("VIGLUMA2", "Glaucoma."),
        ("VIMACDG2", "Macular degeneration."),
        ("CIMEMLOS", "Confusion or memory loss happening more often."),
        ("CDHOUSE", "Memory loss interferes with chores."),
        ("CDASSIST", "Help needed because of memory loss."),
        ("CDHELP", "Able to get help for memory loss."),
        ("CDSOCIAL", "Memory loss interferes with activities outside the home."),
        ("CDDISCUS", "Discussed memory loss with a professional."),
        ("WTCHSALT", "Watching salt intake."),
        ("LONGWTCH", "How long salt has been watched."),
        ("DRADVISE", "Doctor advised reducing salt."),
        ("ASTHMAGE", "Age when first told asthma."),
        ("ASATTACK", "Asthma attack in the past year."),
        ("ASERVIST", "Emergency visit for asthma."),
        ("ASDRVIST", "Urgent doctor visit for asthma."),
        ("ASRCHKUP", "Routine asthma checkup."),
        ("ASACTLIM", "Activities limited by asthma."),
        ("ASYMPTOM", "Asthma symptoms."),
        ("ASNOSLEP", "Days asthma disturbed sleep."),
        ("ASTHMED3", "Asthma medication."),
        ("ASINHALR", "Days an inhaler was used."),
        ("RLIVPAIN", "Limited by joint pain in the arthritis module."),
        ("RDUCHART", "Arthritis advice about reducing heart risk."),
        ("RDUCSTRK", "Arthritis advice about reducing stroke risk."),
        ("ARTTODAY", "Taking medication for arthritis today."),
        ("ARTHWGT", "Advised to lose weight for arthritis."),
        ("ARTHEXER", "Advised to exercise for arthritis."),
        ("ARTHEDU", "Taken a class on arthritis."),
        ("TETANUS", "Tetanus shot, with a packed Tdap code."),
        ("HPVADVC2", "Ever received an HPV vaccination."),
        ("HPVADSHT", "Number of HPV shots."),
        ("SHINGLE2", "Shingles vaccination."),
        ("HADMAM", "Ever had a mammogram."),
        ("HOWLONG", "How long since the last mammogram."),
        ("HADPAP2", "Ever had a Pap test."),
        ("LASTPAP2", "How long since the last Pap test."),
        ("HPVTEST", "Ever had an HPV test."),
        ("HPLSTTST", "How long since the last HPV test."),
        ("HADHYST2", "Had a hysterectomy."),
        ("PROFEXAM", "Clinical breast exam."),
        ("LENGEXAM", "How long since the clinical breast exam."),
        ("BLDSTOOL", "Blood stool test."),
        ("LSTBLDS3", "How long since the blood stool test."),
        ("HADSIGM3", "Ever had a sigmoidoscopy."),
        ("HADSGCO1", "Sigmoidoscopy or colonoscopy."),
        ("LASTSIG3", "How long since that exam."),
        ("PCPSAAD2", "Discussed the PSA test advantages."),
        ("PCPSADI1", "Discussed the PSA test disadvantages."),
        ("PCPSARE1", "Doctor recommended a PSA test."),
        ("PSATEST1", "Ever had a PSA test."),
        ("PSATIME", "How long since the PSA test."),
        ("PCPSARS1", "PSA result discussed."),
        ("PCPSADE1", "Decision about the PSA test."),
        ("PCDMDECN", "Who made the PSA decision."),
        ("SCNTMNY1", "Worried about money for food, social-context module."),
        ("SCNTMEL1", "Worried about meals, social-context module."),
        ("SCNTPAID", "Paid work, social-context module."),
        ("SCNTWRK1", "Work situation, social-context module."),
        ("SCNTLPAD", "Behind on rent or mortgage."),
        ("SCNTLWK1", "Times moved in the past year."),
        ("SXORIENT", "Sexual orientation, optional module."),
        ("TRNSGNDR", "Transgender identity, optional module."),
        ("RCSGENDR", "Gender of sexual partners, optional module."),
        ("RCSRLTN2", "Relationship to sexual partners, optional module."),
        ("CASTHDX2", "Childhood asthma diagnosis, callback module."),
        ("CASTHNO2", "Childhood current asthma, callback module."),
        ("EMTSUPRT", "How often emotional support is received."),
        ("LSATISFY", "Life satisfaction."),
        ("ADPLEASR", "Little interest or pleasure, PHQ item."),
        ("ADDOWN", "Down or depressed, PHQ item."),
        ("ADSLEEP", "Sleep trouble, PHQ item."),
        ("ADENERGY", "Tired, PHQ item."),
        ("ADEAT1", "Appetite, PHQ item."),
        ("ADFAIL", "Feeling like a failure, PHQ item."),
        ("ADTHINK", "Trouble concentrating, PHQ item."),
        ("ADMOVE", "Moving slowly or being fidgety, PHQ item."),
        ("MISTMNT", "Treated badly, optional module."),
        ("ADANXEV", "Ever told an anxiety disorder, optional module."),
    ):
        if name in _SPARSE_KEEP:
            continue
        put(name, "drop", _skip(detail))

    put("BLOODCHO", "drop", _copy(
        "Ever had blood cholesterol checked. Codes 7 and 9 are do not know and refused.",
        "`_CHOLCHK`",
    ))
    put("CHOLCHK", "drop", _copy(
        "How long since cholesterol was checked. The codes are time bins, and "
        "7 and 9 are missing.",
        "`_CHOLCHK`, recoded to checked within five years or not",
    ))
    put("TOLDHI2", "drop", _copy(
        "Ever told cholesterol is high. Codes 7 and 9 are missing.",
        "`_RFCHOL`",
    ))
    put("ASTHMA3", "drop", _copy(
        "Ever told asthma. That is lifetime asthma. Codes 7 and 9 are missing.",
        "`_CASTHM1`, which is current asthma",
    ))
    put("HAVARTH3", "drop", _copy(
        "Ever told arthritis. Codes 7 and 9 are missing.",
        "`_DRDXAR1`",
    ))
    put("EDUCA", "drop", _copy(
        "Education in the raw six-level question. Code 9 is refused.",
        "`_EDUCAG`, the ordered four-level calculated education",
    ))
    put("CHILDREN", "drop", _copy(
        "Number of children. Code 88 means none and 99 means refused, so the "
        "raw number is not a count.",
        "`_CHLDCNT`, recoded to 0 through 5",
    ))
    put("INCOME2", "drop", _copy(
        "Household income in the raw brackets. Codes 77 and 99 are do not know "
        "and refused.",
        "`_INCOMG`",
    ))
    put("WEIGHT2", "drop", _copy(
        "Reported weight. The codes mix pounds, a 9000-plus kilogram scale, "
        "7777 do not know, and 9999 refused. They are not pounds.",
        "`_BMI5`",
    ))
    put("HEIGHT3", "drop", _copy(
        "Reported height. Feet and inches are packed into one integer, and "
        "7777 and 9999 are missing.",
        "`_BMI5`",
    ))
    put("SMOKE100", "drop", _copy(
        "Smoked 100 cigarettes in life. Codes 7 and 9 are missing.",
        "`_SMOKER3`",
    ))
    put("ALCDAY5", "drop", _copy(
        "Days drinking alcohol. Codes 101 to 107 are per week, 201 to 230 per "
        "month, 888 none, 777 do not know, and 999 refused. Left as one number "
        "those are not days.",
        "`_RFBING5` and `_RFDRHV5`",
    ))
    put("FRUITJU1", "drop", _copy(
        "Fruit juice frequency. The codes mix per day, per week, per month, "
        "555 never, 777, and 999.",
        "`_FRTLT1`",
    ))
    put("FRUIT1", "drop", _copy(
        "Fruit frequency, in the same packed codes as `FRUITJU1`.",
        "`_FRTLT1`",
    ))
    put("FVBEANS", "drop", _copy(
        "Bean frequency, in the packed fruit and vegetable codes.",
        "`_VEGLT1`",
    ))
    put("FVGREEN", "drop", _copy(
        "Dark green vegetable frequency, packed codes.",
        "`_VEGLT1`",
    ))
    put("FVORANG", "drop", _copy(
        "Orange vegetable frequency, packed codes.",
        "`_VEGLT1`",
    ))
    put("VEGETAB1", "drop", _copy(
        "Other vegetable frequency, packed codes.",
        "`_VEGLT1`",
    ))
    put("STRENGTH", "drop", _copy(
        "Strength activity frequency, packed like the alcohol day codes, with "
        "888 none and 999 refused.",
        "`_PASTRNG`",
    ))
    put("SEATBELT", "drop", _copy(
        "How often a seat belt is used. The raw codes are not a dose, and 7 "
        "and 9 are missing.",
        "`_RFSEAT3`",
    ))
    put("HIVTST6", "drop", _copy(
        "Ever tested for HIV. Codes 7 and 9 are missing.",
        "`_AIDTST3`",
    ))
    put("INTERNET", "drop", (
        "Internet use in the past 30 days. Codes 7 and 9 are missing. It is "
        "not a health measure and it is not one of the demographic facts kept "
        "with education, income, employment, and race."
    ))
    put("_RFHLTH", "drop", _copy(
        "Calculated fair or poor health from `GENHLTH`. Code 9 is missing.",
        "`GENHLTH`, kept as 1 excellent through 5 poor",
    ))
    put("_HCVU651", "drop", _copy(
        "Health coverage for ages 18 to 64. Code 9 is mostly age 65 and older, "
        "not a refusal, so treating 9 as missing would drop older adults.",
        "`HLTHPLN1`",
    ))
    put("_LTASTH1", "drop", _copy(
        "Calculated lifetime asthma. Code 9 is missing.",
        "`_CASTHM1`, current asthma",
    ))
    put("_ASTHMS1", "drop", _copy(
        "Four-level calculated asthma status. The numbers are categories, and "
        "code 9 is missing.",
        "`_CASTHM1`",
    ))
    put("_CHISPNC", "drop", _copy(
        "Hispanic origin from the child or sample race block. High missing, "
        "and code 9 is refused.",
        "`_RACEGR3`",
    ))
    put("_CRACE1", "drop", _copy(
        "Child or computed race, with 77, 99, and high missing.",
        "`_RACEGR3`",
    ))
    put("_CPRACE", "drop", _copy(
        "Preferred race in the same race block as `_CRACE1`.",
        "`_RACEGR3`",
    ))
    put("_PRACE1", "drop", _copy(
        "Preferred race, with code 99 refused.",
        "`_RACEGR3`",
    ))
    put("_MRACE1", "drop", _copy(
        "Multiracial preferred race, with code 99 refused.",
        "`_RACEGR3`",
    ))
    put("_HISPANC", "drop", _copy(
        "Hispanic origin. Code 9 is refused.",
        "`_RACEGR3`, whose Hispanic level is its own indicator",
    ))
    put("_RACE", "drop", _copy(
        "Detailed race and ethnicity. The codes are categories, not a scale.",
        "`_RACEGR3`",
    ))
    put("_RACEG21", "drop", _copy(
        "White versus all other groups. A coarser copy of race.",
        "`_RACEGR3`",
    ))
    put("_RACE_G1", "drop", _copy(
        "Five-level race with refusals already dropped. Same fact as `_RACEGR3`.",
        "`_RACEGR3`",
    ))
    put("_AGEG5YR", "drop", _copy(
        "Fourteen age bands. Code 9 is a real band in this file, not a refusal, "
        "so it must not be read as missing or as a dose.",
        "`_AGE80`",
    ))
    put("_AGE65YR", "drop", _copy(
        "Under 65 versus 65 and older. Code 3 is the missing age group.",
        "`_AGE80`",
    ))
    put("_AGE_G", "drop", _copy(
        "Six age groups. A coarser copy of age in years.",
        "`_AGE80`",
    ))
    put("HTIN4", "drop", _copy(
        "Height in inches, calculated from `HEIGHT3`.",
        "`_BMI5`",
    ))
    put("HTM4", "drop", _copy("Height in meters, calculated from `HEIGHT3`.", "`_BMI5`"))
    put("WTKG3", "drop", _copy(
        "Weight in kilograms, calculated from `WEIGHT2`.",
        "`_BMI5`",
    ))
    put("_BMI5CAT", "drop", _copy(
        "Four BMI categories of `_BMI5`.",
        "`_BMI5`, the continuous body mass value",
    ))
    put("_RFBMI5", "drop", _copy(
        "Overweight or obese flag. Code 9 is a missing BMI, not a third body size.",
        "`_BMI5`",
    ))
    put("_RFSMOK3", "drop", _copy(
        "Current smoker flag. Code 9 is missing smoking.",
        "`_SMOKER3`, kept as never 0, former 1, some days 2, every day 3",
    ))
    put("DRNKANY5", "drop", _copy(
        "Any alcohol in the past 30 days. Codes 7 and 9 are missing.",
        "`_RFBING5` and `_RFDRHV5`, which already mark non-drinkers as no",
    ))
    put("DROCDY3_", "drop", _copy(
        "Calculated drink occasions, with 900 as a missing sentinel.",
        "`_RFBING5` and `_RFDRHV5`",
    ))
    put("_DRNKWEK", "drop", _copy(
        "Calculated drinks per week, with 99900 as a missing sentinel.",
        "`_RFBING5` and `_RFDRHV5`",
    ))
    for name, detail in (
        ("FTJUDA1_", "Fruit juice times per day."),
        ("FRUTDA1_", "Fruit times per day."),
        ("BEANDAY_", "Bean times per day."),
        ("GRENDAY_", "Green vegetable times per day."),
        ("ORNGDAY_", "Orange vegetable times per day."),
        ("VEGEDA1_", "Other vegetable times per day."),
        ("_FRUTSUM", "Sum of the fruit frequencies. Values can still sit on a 99 hole."),
        ("_VEGESUM", "Sum of the vegetable frequencies."),
    ):
        put(name, "drop", _copy(detail, "`_FRTLT1` and `_VEGLT1`"))
    for name, detail in (
        ("_MISFRTN", "Count of missing fruit items."),
        ("_MISVEGN", "Count of missing vegetable items."),
        ("_FRTRESP", "Answered the fruit questions."),
        ("_VEGRESP", "Answered the vegetable questions."),
        ("_FRT16", "Fruit response bookkeeping flag, nearly constant."),
        ("_VEG23", "Vegetable response bookkeeping flag, nearly constant."),
        ("_FRUITEX", "Fruit exclusion flag."),
        ("_VEGETEX", "Vegetable exclusion flag."),
    ):
        put(name, "drop", (
            detail
            + " Bookkeeping for the fruit and vegetable calculation, not a diet measure."
        ))
    put("_TOTINDA", "drop", _copy(
        "Calculated any leisure activity. Code 9 folds nonresponse into one bin.",
        "`EXERANY2`",
    ))
    for name, detail in (
        ("METVL11_", "MET value of the first activity."),
        ("METVL21_", "MET value of the second activity."),
        (
            "MAXVO2_",
            "Estimated maximum oxygen uptake. Code 999 is missing, and it is not a measured VO2.",
        ),
        ("FC60_", "Estimated functional capacity. Code 999 is missing."),
        ("ACTIN11_", "Aerobic flag for the first activity."),
        ("ACTIN21_", "Aerobic flag for the second activity."),
        ("PADUR1_", "Minutes of the first activity."),
        ("PADUR2_", "Minutes of the second activity."),
        ("PAFREQ1_", "Times per week of the first activity. Code 99 is missing."),
        ("PAFREQ2_", "Times per week of the second activity. Code 99 is missing."),
        ("_MINAC11", "Minutes per week of the first activity."),
        ("_MINAC21", "Minutes per week of the second activity."),
        ("PAMISS1_", "Flag that physical-activity answers are missing."),
        ("PAMIN11_", "Minutes of the first activity after a missing rule."),
        ("PAMIN21_", "Minutes of the second activity after a missing rule."),
        ("PA1MIN_", "Total activity minutes."),
        ("PAVIG11_", "Vigorous-equivalent minutes of the first activity."),
        ("PAVIG21_", "Vigorous-equivalent minutes of the second activity."),
        ("PA1VIGM_", "Total vigorous-equivalent minutes."),
        ("_PACAT1", "Four-level activity category. Code 9 is missing."),
        ("_PAINDX1", "Meets the aerobic guideline. Code 9 is missing."),
        ("_PA150R2", "150-minute activity category. Code 9 is missing."),
        ("_PA300R2", "300-minute activity category. Code 9 is missing."),
        ("_PA30021", "300-minute yes or no. Code 9 is missing."),
        ("_PAREC1", "Activity recommendation category. Code 9 is missing."),
        ("_PASTAE1", "Meets the aerobic and strength guideline. Code 9 is missing."),
    ):
        put(name, "drop", _copy(
            detail + " Derived from the same activity answers.",
            "`EXERANY2` for any activity and `_PASTRNG` for strength",
        ))
    put("STRFREQ_", "drop", _copy(
        "Strength sessions per week. Code 99 is missing.",
        "`_PASTRNG`",
    ))
    put("_LMTACT1", "drop", (
        "Calculated limitation of usual activities by arthritis. Code 3 means "
        "no arthritis, which is the majority, so a linear reading treats no "
        "arthritis as a high score. The diagnosis is kept as `_DRDXAR1`. "
        "Limitation for the whole sample is kept as `DIFFWALK` and `QLACTLM2`."
    ))
    put("_LMTWRK1", "drop", (
        "Calculated work limitation from arthritis, with the same non-linear "
        "code 3 for no arthritis. `_DRDXAR1`, `DIFFWALK`, and `QLACTLM2` are kept."
    ))
    put("_LMTSCL1", "drop", (
        "Calculated social limitation from arthritis. The codes are categories, "
        "and code 4 is no arthritis. `_DRDXAR1` is kept."
    ))
    put("_RFSEAT2", "drop", _copy(
        "Usually wears a seat belt, an alternate calculated cut. Code 9 is missing.",
        "`_RFSEAT3`",
    ))

    put("GENHLTH", "keep", (
        "General health. Keep 1 excellent through 5 poor. Codes 7 and 9 become "
        "missing before the mode fill. `_RFHLTH` is the dropped copy."
    ))
    put("PHYSHLTH", "keep", (
        "Days in the past 30 days when physical health was not good. Code 88 "
        "means zero days, not 88 days, so it is recoded to 0. Codes 77 and 99 "
        "become missing. Codes 7 and 9 stay, because they are seven and nine "
        "days, not do-not-know codes."
    ))
    put("MENTHLTH", "keep", (
        "Days in the past 30 days when mental health was not good. Same recode "
        "as `PHYSHLTH`: 88 becomes 0, 77 and 99 become missing, and 7 and 9 "
        "stay as day counts."
    ))
    put("HLTHPLN1", "keep", (
        "Has any health coverage. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("PERSDOC2", "keep", (
        "Has a personal doctor. Codes 1 and 2, one doctor or more than one, "
        "become 1. Code 3, none, becomes 0. Codes 7 and 9 become missing."
    ))
    put("MEDCOST", "keep", (
        "Could not see a doctor because of cost. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing."
    ))
    put("CHECKUP1", "keep", (
        "Time since the last checkup. Codes 1 through 4 stay in that order, "
        "from within a year to five or more years. Code 8, never, becomes 5 "
        "so it continues that order. Codes 7 and 9 become missing."
    ))
    put("BPMEDS", "keep", (
        "Takes blood pressure medication. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. If `BPHIGH4` is 2, 3, or 4 the person "
        "is not a treated hypertensive, so this is set to 0 and is not left "
        "for the mode fill. `BPHIGH4` is then dropped."
    ))
    put("_RFHYPE5", "keep", (
        "Calculated high blood pressure. CDC uses 2 for yes and 1 for no. "
        "Yes becomes 1, no becomes 0, and 9 becomes missing. Pregnancy-only "
        "and borderline answers sit in that 9."
    ))
    put("_CHOLCHK", "keep", (
        "Cholesterol check. Code 1, within five years, becomes 1. Codes 2 and "
        "3, older check or never, become 0. Code 9 becomes missing."
    ))
    put("_RFCHOL", "keep", (
        "Calculated high cholesterol. CDC uses 2 for yes and 1 for no. Yes "
        "becomes 1, no becomes 0, and 9 becomes missing. `TOLDHI2` is the dropped copy. "
        "`rfchol_missing` is 1 on those missing rows, so the mode fill does not "
        "hide the refusal."
    ))
    put("CVDSTRK3", "keep", (
        "Ever told stroke. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and "
        "9 become missing."
    ))
    put("_CASTHM1", "keep", (
        "Current asthma. This calculated item uses 2 for yes and 1 for no. "
        "Yes becomes 1, no becomes 0, and 9 becomes missing. Lifetime asthma "
        "`ASTHMA3` is the dropped copy."
    ))
    put("CHCSCNCR", "keep", (
        "Ever told skin cancer. Code 1 becomes 1 and code 2 becomes 0. Codes "
        "7 and 9 become missing."
    ))
    put("CHCOCNCR", "keep", (
        "Ever told another cancer. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. Kept separately from skin cancer."
    ))
    put("CHCCOPD1", "keep", (
        "Ever told COPD, emphysema, or chronic bronchitis. Code 1 becomes 1 "
        "and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("_DRDXAR1", "keep", (
        "Doctor-diagnosed arthritis. Code 1 becomes 1 and code 2 becomes 0. "
        "`HAVARTH3` is the dropped copy."
    ))
    put("ADDEPEV2", "keep", (
        "Ever told a depressive disorder. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. The PHQ day counts are kept separately."
    ))
    put("CHCKIDNY", "keep", (
        "Ever told kidney disease. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("DIABETE3", "keep", (
        "Diabetes is not one increasing scale. Three indicators are built, "
        "with code 3, no diabetes, as the reference of all zeros: `diab_yes` "
        "for code 1, `diab_pregnancy` for code 2, and `diab_pre` for code 4. "
        "Codes 7 and 9 make all three missing. The age product uses `diab_yes` only."
    ))
    put("QLACTLM2", "keep", (
        "Limited in any activities because of a health problem. Code 1 becomes "
        "1 and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("USEEQUIP", "keep", (
        "Health problem requires special equipment. Code 1 becomes 1 and code "
        "2 becomes 0. Codes 7 and 9 become missing. Distinct from difficulty walking."
    ))
    put("BLIND", "keep", (
        "Blind or serious difficulty seeing. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing."
    ))
    put("DECIDE", "keep", (
        "Serious difficulty concentrating or making decisions. Code 1 becomes "
        "1 and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("DIFFWALK", "keep", (
        "Serious difficulty walking or climbing stairs. Code 1 becomes 1 and "
        "code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("DIFFDRES", "keep", (
        "Difficulty dressing or bathing. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("DIFFALON", "keep", (
        "Difficulty doing errands alone. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("_SMOKER3", "keep", (
        "Calculated smoking. Every day, code 1, becomes 3. Some days, code 2, "
        "becomes 2. Former, code 3, becomes 1. Never, code 4, becomes 0. "
        "Code 9 becomes missing. `SMOKE100` and `SMOKDAY2` are the dropped copies."
    ))
    put("USENOW3", "keep", (
        "Smokeless tobacco, a different product from cigarettes. Every day "
        "becomes 2, some days becomes 1, and not at all becomes 0. Codes 7 "
        "and 9 become missing."
    ))
    put("EXERANY2", "keep", (
        "Any physical activity in the past month. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing. `_TOTINDA` is the dropped copy. "
        "The activity-minute calculations are not kept."
    ))
    put("_PASTRNG", "keep", (
        "Muscle strengthening two or more times a week. Code 1 becomes 1 and "
        "code 2 becomes 0. Code 9 becomes missing. The raw `STRENGTH` frequency "
        "is the dropped copy. `pastrng_missing` marks that hole before the mode fill."
    ))
    put("_RFBING5", "keep", (
        "Calculated binge drinking. CDC uses 2 for yes and 1 for no. Yes "
        "becomes 1, no becomes 0, and 9 becomes missing."
    ))
    put("_RFDRHV5", "keep", (
        "Calculated heavy drinking, which is not the same fact as binge "
        "drinking. CDC uses 2 for yes and 1 for no. Yes becomes 1, no becomes "
        "0, and 9 becomes missing."
    ))
    put("_FRTLT1", "keep", (
        "Fruit less than once a day. Code 1 becomes 1 and code 2 becomes 0. "
        "Code 9 becomes missing. The raw frequency items are dropped because "
        "their packed codes are not counts. `frtlt1_missing` marks that hole "
        "before the mode fill."
    ))
    put("_VEGLT1", "keep", (
        "Vegetables less than once a day. Code 1 becomes 1 and code 2 becomes "
        "0. Code 9 becomes missing. `veglt1_missing` marks that hole before the mode fill."
    ))
    put("_RFSEAT3", "keep", (
        "Always or nearly always wears a seat belt. Code 1 becomes 1 and code "
        "2 becomes 0. Code 9 becomes missing. `rfseat3_missing` marks that hole "
        "before the mode fill."
    ))
    put("_AIDTST3", "keep", (
        "Ever tested for HIV. Code 1 becomes 1 and code 2 becomes 0. Code 9 "
        "becomes missing. `HIVTST6` is the dropped copy. `aidtst3_missing` "
        "marks that hole before the mode fill."
    ))
    put("SEX", "keep", (
        "Sex. Code 1, male, stays 1. Code 2, female, becomes 0. The file has "
        "no refused code on this item."
    ))
    put("_AGE80", "keep", (
        "Age in years. Values 18 through 79 stay as years, and 80 means 80 or "
        "older. The square of this column is added later. `_AGEG5YR`, "
        "`_AGE65YR`, and `_AGE_G` are the dropped copies."
    ))
    put("_BMI5", "keep", (
        "Body mass index. In this file the values are already body mass units, "
        "such as 26.6, not the CDC integer times 100. Blanks are the missing "
        "values. There is no 9999 sentinel here, and a large value such as 77 "
        "is a BMI, not a do-not-know code. Values above 60 are set to 60 before "
        "the square, so a few extreme rows do not dominate `bmi5_sq`. "
        "`WEIGHT2` and `HEIGHT3` are the dropped copies."
    ))
    put("_EDUCAG", "keep", (
        "Calculated education, kept as the ordered codes 1 through 4. Higher "
        "means more education. Code 9 becomes missing. `EDUCA` is the dropped copy."
    ))
    put("_INCOMG", "keep", (
        "Calculated income, kept as the ordered codes 1 through 5. Higher "
        "means more income. Code 9 becomes missing. `INCOME2` is the dropped copy. "
        "`incomg_missing` is 1 on those rows. The mode alone would mark a refusal "
        "as the top income bracket."
    ))
    put("_CHLDCNT", "keep", (
        "Number of children. CDC code 1 means none, so the codes are shifted "
        "to 0 through 5, with 5 meaning five or more. Code 9 becomes missing. "
        "`CHILDREN` is the dropped copy, because 88 there means none."
    ))
    put("VETERAN3", "keep", (
        "Veteran. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("EMPLOY1", "keep", (
        "Employment is a category, not one linear number. The reference, all "
        "zeros, is code 1, employed for wages. Indicators are self-employed, "
        "out of work a year or more, out of work less than a year, homemaker, "
        "student, retired, and unable to work. Code 7 is retired, a real "
        "category, not do not know. Code 9 becomes missing on every indicator. "
        "A refusal is then mode filled, which usually becomes the reference."
    ))
    put("MARITAL", "keep", (
        "Marital status is a category. The reference is code 1, married. "
        "Indicators are divorced, widowed, separated, never married, and "
        "unmarried couple. Code 9 becomes missing on every indicator."
    ))
    put("RENTHOM1", "keep", (
        "Home ownership is a category. The reference is code 1, own. Indicators "
        "are rent and other arrangement. Codes 7 and 9 become missing. This is "
        "not the same fact as the income bracket."
    ))
    put("_RACEGR3", "keep", (
        "Race and ethnicity are categories. The reference is code 1, white "
        "non-Hispanic. Indicators are black non-Hispanic, other race "
        "non-Hispanic, multiracial non-Hispanic, and Hispanic. Code 9 becomes "
        "missing on every indicator. The other race columns are dropped copies."
    ))
    _put_sparse_keeps(put)

    missing = [name for name in HEADER if name not in reasons]
    extra = [name for name in reasons if name not in INDEX]
    if missing or extra:
        raise RuntimeError("decision coverage missing=%s extra=%s" % (missing, extra))
    return reasons


def _put_sparse_keeps(put):
    """Keep health items that are blank for most rows.

    The blank is not a measured zero. Training adds a missing indicator
    before the mode fill, and the preprocessor does not drop the column
    for being mostly empty.
    """
    put("POORHLTH", "keep", (
        "Days poor health limited activities. Asked only when physical or "
        "mental health was not good, so most rows are blank. Codes 1 through "
        "30 stay, 88 becomes 0 days, and 77 and 99 become missing. The blank "
        "gets a missing indicator before the mode fill."
    ))
    put("LMTJOIN3", "keep", (
        "Limited by joint symptoms. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. Most rows were not asked. The missing "
        "indicator keeps that fact."
    ))
    put("ARTHDIS2", "keep", (
        "Arthritis limits usual activities. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing. A blank is not a zero."
    ))
    put("ARTHSOCL", "keep", (
        "Arthritis limits social activities. Codes 1 a lot, 2 a little, and "
        "3 not at all stay in that order. Codes 7 and 9 become missing."
    ))
    put("JOINPAIN", "keep", (
        "Joint pain on a 0 to 10 scale. Those scores stay. Codes 77 and 99 "
        "become missing. A blank means the arthritis pain item was not asked."
    ))
    put("ARTHWGT", "keep", (
        "Advised to lose weight for arthritis. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing."
    ))
    put("ARTHEXER", "keep", (
        "Advised to exercise for arthritis. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing."
    ))
    put("CIMEMLOS", "keep", (
        "Confusion or memory loss happening more often. Code 1 becomes 1 and "
        "code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("PDIABTST", "keep", (
        "Tested for high blood sugar in the past three years. Code 1 becomes "
        "1 and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("INSULIN", "keep", (
        "Now taking insulin. Asked of people with diabetes. Code 1 becomes 1 "
        "and code 2 becomes 0. Codes 7 and 9 become missing."
    ))
    put("DIABEYE", "keep", (
        "Told diabetes affected the eyes. Code 1 becomes 1 and code 2 becomes "
        "0. Codes 7 and 9 become missing."
    ))
    put("DIABEDU", "keep", (
        "Taken a class on managing diabetes. Code 1 becomes 1 and code 2 "
        "becomes 0. Codes 7 and 9 become missing."
    ))
    put("DIABAGE2", "keep", (
        "Age when told diabetes. Ages 1 through 97 stay. Codes 98 and 99 "
        "become missing. A blank is everyone who was not asked, not age zero."
    ))
    put("CHKHEMO3", "keep", (
        "Times A1C was checked. Counts 1 through 76 stay, and 88, none, "
        "becomes 0. Codes 77, 98, and 99 become missing."
    ))
    put("DOCTDIAB", "keep", (
        "Times a professional was seen for diabetes. Counts 1 through 76 stay, "
        "and 88, none, becomes 0. Codes 77 and 99 become missing."
    ))
    put("EYEEXAM", "keep", (
        "Last dilated eye exam among people with diabetes. Codes 1 through 4 "
        "stay, from the past month to two or more years. Code 8, never, "
        "becomes 5. Codes 7 and 9 become missing."
    ))
    put("WTCHSALT", "keep", (
        "Watching salt intake. Code 1 becomes 1 and code 2 becomes 0. Codes "
        "7 and 9 become missing."
    ))
    put("DRADVISE", "keep", (
        "Doctor advised reducing salt. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("AVEDRNK2", "keep", (
        "Average drinks when drinking. Counts 1 through 76 stay. Codes 77 and "
        "99 become missing. Asked only of people who drank."
    ))
    put("DRNK3GE5", "keep", (
        "Binge occasions among drinkers. Counts 1 through 76 stay, and 88, "
        "none, becomes 0. Codes 77 and 99 become missing."
    ))
    put("MAXDRNKS", "keep", (
        "Largest number of drinks on one occasion. Counts 1 through 76 stay. "
        "Codes 77 and 99 become missing."
    ))
    put("STOPSMK2", "keep", (
        "Stopped smoking for one day or longer in the past year. Asked of "
        "current smokers. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and "
        "9 become missing."
    ))
    put("FLUSHOT6", "keep", (
        "Flu shot in the past year. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. A blank means the question was not asked."
    ))
    put("PNEUVAC3", "keep", (
        "Pneumonia vaccination. Code 1 becomes 1 and code 2 becomes 0. Codes "
        "7 and 9 become missing. A blank means the question was not asked."
    ))
    put("_FLSHOT6", "keep", (
        "Calculated flu shot. Code 1, yes, stays 1. Code 2, no, becomes 0. "
        "Code 9 becomes missing. Younger adults are often unscored, and that "
        "hole is a missing indicator rather than a dropped column."
    ))
    put("_PNEUMO2", "keep", (
        "Calculated pneumonia vaccination. Code 1 stays 1, code 2 becomes 0, "
        "and code 9 becomes missing. Same age hole as `_FLSHOT6`."
    ))
    put("SHINGLE2", "keep", (
        "Shingles vaccination. Code 1 becomes 1 and code 2 becomes 0. Codes "
        "7 and 9 become missing."
    ))
    put("BLDSTOOL", "keep", (
        "Blood stool test. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and "
        "9 become missing."
    ))
    put("HADSIGM3", "keep", (
        "Ever had a sigmoidoscopy. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing."
    ))
    put("ADANXEV", "keep", (
        "Ever told an anxiety disorder. Code 1 becomes 1 and code 2 becomes 0. "
        "Codes 7 and 9 become missing. This optional module is mostly blank."
    ))
    put("CAREGIV1", "keep", (
        "Provided care to a friend or family member. Code 1 becomes 1 and "
        "code 2 becomes 0. Codes 7, 8, and 9 become missing."
    ))
    put("PREGNANT", "keep", (
        "Currently pregnant. Code 1 becomes 1 and code 2 becomes 0. Codes 7 "
        "and 9 become missing. Men, sex code 1, are set to 0 here, because "
        "the question was not asked of them. Women who were not asked stay "
        "missing for the mode fill."
    ))
    put("EMTSUPRT", "keep", (
        "How often emotional support is received. Codes 1 always through 5 "
        "never stay in that order. Codes 7 and 9 become missing."
    ))
    put("LSATISFY", "keep", (
        "Life satisfaction. Codes 1 very satisfied through 4 very dissatisfied "
        "stay. Codes 7 and 9 become missing."
    ))
    put("SCNTMNY1", "keep", (
        "How often money for housing was a worry. Codes 1 always through 5 "
        "never stay. Codes 7 and 9 become missing."
    ))
    put("SCNTMEL1", "keep", (
        "How often money for meals was a worry. Codes 1 always through 5 "
        "never stay. Codes 7 and 9 become missing."
    ))
    for name, detail in (
        ("ADPLEASR", "Little interest or pleasure."),
        ("ADDOWN", "Down or depressed."),
        ("ADSLEEP", "Sleep trouble."),
        ("ADENERGY", "Tired."),
        ("ADEAT1", "Appetite."),
        ("ADFAIL", "Feeling like a failure."),
        ("ADTHINK", "Trouble concentrating."),
        ("ADMOVE", "Moving slowly or being fidgety."),
    ):
        put(name, "keep", (
            detail
            + " Days 1 through 14 stay, 88 becomes 0 days, and 77 and 99 "
            + "become missing. The PHQ module is blank for most rows."
        ))


DECISIONS = _build_reasons()


def _check_plan():
    planned = [name for name, _kind in PLAN]
    if len(planned) != len(set(planned)):
        raise RuntimeError("PLAN repeats a column")
    kept = {name for name, (action, _text) in DECISIONS.items() if action == "keep"}
    if set(planned) != kept:
        raise RuntimeError("PLAN and keep decisions differ")
    for name, kind in PLAN:
        if kind == "indicators" and name not in _INDICATORS:
            raise RuntimeError(name + " has no indicators")
        if kind == "map" and _ORDINAL.get(name) is None:
            raise RuntimeError(name + " has no map")
        if kind == "days" and name not in ("PHYSHLTH", "MENTHLTH"):
            raise RuntimeError(name)
        if kind == "bounded" and name not in _BOUNDED:
            raise RuntimeError(name + " has no bounds")
        if kind == "pregnant" and name != "PREGNANT":
            raise RuntimeError(name)
        if kind == "age" and name != "_AGE80":
            raise RuntimeError(name)
        if kind == "bmi" and name != "_BMI5":
            raise RuntimeError(name)
        if kind == "bpmeds" and name != "BPMEDS":
            raise RuntimeError(name)
    for source_name, _flag_name in MISSING_FLAGS:
        if source_name not in kept:
            raise RuntimeError(source_name + " is not a kept column")


_check_plan()


def flag_name_for(column_name):
    """Missing-indicator name for one recoded column."""
    return column_name.lower().lstrip("_") + "_missing"


def choose_flag_names(values, names, min_rate, always=None):
    """Output names that receive a missing indicator.

    `always` is flagged even below `min_rate`. A later column whose missing
    pattern matches an earlier flag is skipped, so one skipped module does
    not add several copies of the same hole. Fit this on the training matrix
    and pass the same names to test.
    """
    matrix = np.asarray(values, dtype=float)
    if always is None:
        always = [source for source, _flag in MISSING_FLAGS]
    rates = np.isnan(matrix).mean(axis=0)
    index_of = {name: index for index, name in enumerate(names)}
    wanted = []
    seen = set()
    for name in list(always) + list(names):
        if name in seen or name not in index_of:
            continue
        if name not in always and rates[index_of[name]] < min_rate:
            continue
        seen.add(name)
        wanted.append(name)
    chosen = []
    seen_masks = set()
    for name in wanted:
        mask = np.isnan(matrix[:, index_of[name]])
        key = mask.tobytes()
        if key in seen_masks:
            continue
        seen_masks.add(key)
        chosen.append(name)
    return chosen


def transform_codebook(x, flag_names=None):
    """Recode one matrix. Does not impute and does not standardize.

    Input: x (N, 321) after `Id` is removed. NaN is already missing.
    `flag_names` lists recoded columns that get a missing indicator.
    None uses the seven named holes in `MISSING_FLAGS`. An empty sequence
    adds no flags.
    Output: values (N, K), names length K, source_index (K,) into `x`.
    """
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(HEADER):
        raise ValueError(
            "expected %d columns after Id is removed, got shape %s"
            % (len(HEADER), matrix.shape)
        )
    columns = []
    names = []
    sources = []
    for source_name, kind in PLAN:
        built = _apply(matrix, source_name, kind)
        for output_name, values in built:
            columns.append(values)
            names.append(output_name)
            sources.append(INDEX[source_name])
    if flag_names is None:
        flag_names = [source for source, _flag in MISSING_FLAGS]
    for source_name in flag_names:
        position = names.index(source_name)
        columns.append(np.isnan(columns[position]).astype(float))
        names.append(flag_name_for(source_name))
        sources.append(sources[position])
    return np.column_stack(columns), names, np.asarray(sources, dtype=int)


def columns_after_drop(names, source_index, drop):
    """Drop codebook columns the preprocessor rejected.

    Input: names length K, source_index (K,), drop (K,) bool from
    `fit_preprocessor`.
    Output: kept names, kept source indices.
    """
    mask = np.asarray(drop, dtype=bool)
    if mask.shape[0] != len(names):
        raise ValueError("drop mask does not match codebook columns")
    kept = ~mask
    kept_names = [name for name, flag in zip(names, kept) if flag]
    return kept_names, np.asarray(source_index)[kept]


def render_feature_choices():
    """Markdown with one paragraph per original column, in file order."""
    lines = [
        "# Feature choices, BRFSS 2015",
        "",
        "The label is self-reported heart attack or coronary heart disease "
        "(`_MICHD`). Labels in the course files are `{-1, +1}`. About 8.8% "
        "are positive. `CVDINFR4` and `CVDCRHD4` are not in this extract, and "
        "`_MICHD` is not reconstructed.",
        "",
        "Index is the 0-based column in the array from `load_csv_data`, after "
        "`Id` is removed. `FMONTH` is 1, `IYEAR` is 5, and `WEIGHT2` is 62.",
        "",
        "Order of operations. Recode from this file first. Do-not-know and "
        "refused codes become missing. Sparse health items stay in the matrix. "
        "A blank is not treated as a measured zero. Training then adds a "
        "missing indicator, before the mode fill, for every recoded column "
        "whose training missing rate is at least `missing_flag_min_rate` in "
        "`config/features.json`, and always for income, high cholesterol, "
        "fruit, vegetables, seat belt, strength training, and the HIV test. "
        "The mode fill, the constant-column drop, and standardization come "
        "next. Columns are not dropped for being mostly empty. That cutoff "
        "is `max_missing_rate` in the same file, set so a high missing rate "
        "is kept. `_BMI5` values above the configured cap are set to that cap. "
        "Then write the leave-one-column partial R^2 to `column_significance.csv`. "
        "That score is a report. It does not drop columns and it cannot bring "
        "`IYEAR` back, because `IYEAR` is already gone. Then add the squares "
        "of age and BMI, and the products of age with sex, smoking, diagnosed "
        "diabetes, general health, and high cholesterol, plus BMI with smoking.",
        "",
        "On indicator blocks, a refused row is missing on every indicator. "
        "The later mode fill usually turns that into the reference level, "
        "which is the all-zero pattern. `BPMEDS` does not use that rule for "
        "people who are not treated hypertensives. Those rows are set to 0 "
        "here.",
        "",
        "Only the standard library, NumPy, and the plotting libraries are used. "
        "The six functions in `implementations.py` are unchanged.",
        "",
    ]
    for index, name in enumerate(HEADER):
        action, text = DECISIONS[name]
        lines.append("**%d `%s` — %s.** %s" % (index, name, action, text))
        lines.append("")
    lines.append("## Nonlinear terms")
    lines.append("")
    lines.append(
        "These are not original columns. They are added after standardization "
        "of the codebook matrix, then standardized with it."
    )
    lines.append("")
    lines.append(
        "`age80_sq` is the square of `_AGE80`. `bmi5_sq` is the square of "
        "capped `_BMI5`. `age_x_sex`, `age_x_smoker`, and `age_x_diab_yes` "
        "are `_AGE80` times `SEX`, `_SMOKER3`, and `diab_yes`. "
        "`age_x_genhlth` and `age_x_rfchol` are `_AGE80` times `GENHLTH` and "
        "`_RFCHOL`. `bmi_x_smoker` is `_BMI5` times `_SMOKER3`. "
        "Pregnancy-only diabetes and prediabetes stay linear. Logistic "
        "regression receives these same columns. The sigmoid bends the score. "
        "It does not create these products."
    )
    lines.append("")
    lines.append("## Model column order")
    lines.append("")
    _dummy = np.zeros((1, len(HEADER)))
    _values, names, _sources = transform_codebook(_dummy)
    lines.append(
        "The codebook matrix has %d columns, in this order. Squares and "
        "products are added after standardization and are not in this list."
        % _values.shape[1]
    )
    lines.append("")
    for index, name in enumerate(names):
        lines.append("%d. `%s`" % (index, name))
    lines.append("")
    wrapped = []
    for line in lines:
        heading = line.startswith("#") or line[:1].isdigit()
        if line == "" or heading:
            wrapped.append(line)
        else:
            wrapped.append(
                textwrap.fill(
                    line, width=100, break_long_words=False, break_on_hyphens=False
                )
            )
    return "\n".join(wrapped)


def write_feature_choices(path):
    """Write `render_feature_choices` to `path`."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_feature_choices())


def _apply(matrix, source_name, kind):
    source = matrix[:, INDEX[source_name]]
    if kind == "map":
        return ((source_name, _map_values(source, _ORDINAL[source_name])),)
    if kind == "days":
        return ((source_name, _health_days(source)),)
    if kind == "age":
        return ((source_name, _age80(source)),)
    if kind == "bmi":
        return ((source_name, _bmi(source)),)
    if kind == "indicators":
        levels = _INDICATORS[source_name]
        arrays = _indicators(source, levels, _INDICATOR_VALID[source_name])
        return tuple((name, values) for (_code, name), values in zip(levels, arrays))
    if kind == "bpmeds":
        blood_pressure = matrix[:, INDEX["BPHIGH4"]]
        return (("BPMEDS", _bpmeds(blood_pressure, source)),)
    if kind == "bounded":
        low, high, zero_codes = _BOUNDED[source_name]
        return ((source_name, _bounded(source, low, high, zero_codes)),)
    if kind == "pregnant":
        sex = matrix[:, INDEX["SEX"]]
        return (("PREGNANT", _pregnant(sex, source)),)
    raise ValueError("unknown recode kind: " + kind)


def _map_values(column, mapping):
    """Map listed codes and set every other finite or missing code to NaN."""
    out = np.full(column.shape[0], np.nan, dtype=float)
    for code, value in mapping.items():
        out[column == code] = value
    return out


def _health_days(column):
    """Days 1 through 30 stay. Code 88 becomes 0. Codes 77 and 99 stay missing."""
    out = np.full(column.shape[0], np.nan, dtype=float)
    days = (column >= 1) & (column <= 30)
    out[days] = column[days]
    out[column == 88] = 0.0
    return out


def _age80(column):
    """Keep ages 18 through 80. Code 80 already means 80 or older."""
    out = np.full(column.shape[0], np.nan, dtype=float)
    ok = (column >= 18) & (column <= 80)
    out[ok] = column[ok]
    return out


def _bmi(column):
    """Keep body mass units, and cap the extreme tail at `BMI_CAP`.

    Non-finite cells stay missing. The cap is applied before standardization
    and before the square.
    """
    out = column.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    above = out > BMI_CAP
    out[above] = BMI_CAP
    return out


def _indicators(column, levels, valid):
    """One column per non-reference code. The reference is all zeros.

    Unknown codes, including 7 and 9, are missing on every indicator so the
    mode fill does not mix a refusal into a single level at random.
    """
    known = np.isin(column, valid)
    built = []
    for code, _name in levels:
        out = np.full(column.shape[0], np.nan, dtype=float)
        out[known] = (column[known] == code).astype(float)
        built.append(out)
    return built


def _bounded(column, low, high, zero_codes):
    """Keep values from `low` through `high`. Listed codes become 0.

    Every other code, including do-not-know and refused, stays missing.
    """
    out = np.full(column.shape[0], np.nan, dtype=float)
    ok = (column >= low) & (column <= high)
    out[ok] = column[ok]
    for code in zero_codes:
        out[column == code] = 0.0
    return out


def _pregnant(sex, column):
    """Pregnancy 1 or 0. Men are 0, because the question was not asked."""
    out = _map_values(column, {1: 1, 2: 0})
    out[sex == 1] = 0.0
    return out


def _bpmeds(blood_pressure, medication):
    """Medication 1 or 0, with non-hypertensives forced to 0.

    `BPHIGH4` codes 2, 3, and 4 are pregnancy-only, no, and borderline.
    Those rows are 0 even when the medication item is blank. Codes 7 and 9
    on the medication item stay missing when blood pressure was reported as
    high. Unknown blood pressure also stays missing.
    """
    out = np.full(blood_pressure.shape[0], np.nan, dtype=float)
    out[medication == 1] = 1.0
    out[medication == 2] = 0.0
    not_treated = np.isin(blood_pressure, (2, 3, 4))
    out[not_treated] = 0.0
    return out
