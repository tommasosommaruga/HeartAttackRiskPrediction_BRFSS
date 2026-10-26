# Feature choices, BRFSS 2015

The label is self-reported heart attack or coronary heart disease (`_MICHD`). Labels in the course
files are `{-1, +1}`. About 8.8% are positive. `CVDINFR4` and `CVDCRHD4` are not in this extract,
and `_MICHD` is not reconstructed.

Index is the 0-based column in the array from `load_csv_data`, after `Id` is removed. `FMONTH` is 1,
`IYEAR` is 5, and `WEIGHT2` is 62.

Order of operations. Recode from this file first. Do-not-know and refused codes become missing.
Sparse health items stay in the matrix. A blank is not treated as a measured zero. Training then
adds a missing indicator, before the mode fill, for every recoded column whose training missing rate
is at least `missing_flag_min_rate` in `config/features.json`, and always for income, high
cholesterol, fruit, vegetables, seat belt, strength training, and the HIV test. The mode fill, the
constant-column drop, and standardization come next. Columns are not dropped for being mostly empty.
That cutoff is `max_missing_rate` in the same file, set so a high missing rate is kept. `_BMI5`
values above the configured cap are set to that cap. Then write the leave-one-column partial R^2 to
`column_significance.csv`. That score is a report. It does not drop columns and it cannot bring
`IYEAR` back, because `IYEAR` is already gone. Then add the squares of age and BMI, and the products
of age with sex, smoking, diagnosed diabetes, general health, and high cholesterol, plus BMI with
smoking.

On indicator blocks, a refused row is missing on every indicator. The later mode fill usually turns
that into the reference level, which is the all-zero pattern. `BPMEDS` does not use that rule for
people who are not treated hypertensives. Those rows are set to 0 here.

Only the standard library, NumPy, and the plotting libraries are used. The six functions in
`implementations.py` are unchanged.

**0 `_STATE` — drop.** State FIPS code. It is the sampling geography. As one number the codes are
not an ordered exposure. Survey machinery, not a health measure. Removed before the column score, so
the score cannot put it back.

**1 `FMONTH` — drop.** Interview month, 1 through 12. It is the survey calendar. Survey machinery,
not a health measure. Removed before the column score, so the score cannot put it back.

**2 `IDATE` — drop.** Interview date packed as a month-day-year integer. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**3 `IMONTH` — drop.** Interview month, a second copy of the calendar. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**4 `IDAY` — drop.** Interview day of the month. Survey machinery, not a health measure. Removed
before the column score, so the score cannot put it back.

**5 `IYEAR` — drop.** Interview year. Almost every row is 2015, and the rest are 2016 fieldwork
spillover. It is not age or disease. Survey machinery, not a health measure. Removed before the
column score, so the score cannot put it back.

**6 `DISPCODE` — drop.** Disposition code, 1100 completed or 1200 partial. It records how the
interview ended. Survey machinery, not a health measure. Removed before the column score, so the
score cannot put it back.

**7 `SEQNO` — drop.** Sample sequence number. It identifies the interview. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**8 `_PSU` — drop.** Primary sampling unit. In this file it matches the sequence number. Survey
machinery, not a health measure. Removed before the column score, so the score cannot put it back.

**9 `CTELENUM` — drop.** Landline phone number is correct. A phone check. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**10 `PVTRESD1` — drop.** Private residence flag for the landline sample. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**11 `COLGHOUS` — drop.** College housing flag. Asked only in a tiny skip. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**12 `STATERES` — drop.** Resident of the sampled state. A sample check. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**13 `CELLFON3` — drop.** Has a cell phone. A landline sample question. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**14 `LADULT` — drop.** Adult in the landline household. A sample check. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**15 `NUMADULT` — drop.** Number of adults in the landline household. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**16 `NUMMEN` — drop.** Number of adult men in the landline household. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**17 `NUMWOMEN` — drop.** Number of adult women in the landline household. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**18 `CTELNUM1` — drop.** Cell phone number is correct. A phone check. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**19 `CELLFON2` — drop.** Cell-sample interview flag. Survey machinery, not a health measure.
Removed before the column score, so the score cannot put it back.

**20 `CADULT` — drop.** Adult on the cell phone. A sample check. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**21 `PVTRESD2` — drop.** Private residence flag for the cell sample. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**22 `CCLGHOUS` — drop.** College housing flag on the cell sample. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**23 `CSTATE` — drop.** Cell respondent lives in the sampled state. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**24 `LANDLINE` — drop.** Has a landline. Codes include 7 and 9, and the item is a sample field.
Survey machinery, not a health measure. Removed before the column score, so the score cannot put it
back.

**25 `HHADULT` — drop.** Adults in the cell-phone household. Code 99 is refused, and the item is a
sample field. Survey machinery, not a health measure. Removed before the column score, so the score
cannot put it back.

**26 `GENHLTH` — keep.** General health. Keep 1 excellent through 5 poor. Codes 7 and 9 become
missing before the mode fill. `_RFHLTH` is the dropped copy.

**27 `PHYSHLTH` — keep.** Days in the past 30 days when physical health was not good. Code 88 means
zero days, not 88 days, so it is recoded to 0. Codes 77 and 99 become missing. Codes 7 and 9 stay,
because they are seven and nine days, not do-not-know codes.

**28 `MENTHLTH` — keep.** Days in the past 30 days when mental health was not good. Same recode as
`PHYSHLTH`: 88 becomes 0, 77 and 99 become missing, and 7 and 9 stay as day counts.

**29 `POORHLTH` — keep.** Days poor health limited activities. Asked only when physical or mental
health was not good, so most rows are blank. Codes 1 through 30 stay, 88 becomes 0 days, and 77 and
99 become missing. The blank gets a missing indicator before the mode fill.

**30 `HLTHPLN1` — keep.** Has any health coverage. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing.

**31 `PERSDOC2` — keep.** Has a personal doctor. Codes 1 and 2, one doctor or more than one, become
1. Code 3, none, becomes 0. Codes 7 and 9 become missing.

**32 `MEDCOST` — keep.** Could not see a doctor because of cost. Code 1 becomes 1 and code 2 becomes
0. Codes 7 and 9 become missing.

**33 `CHECKUP1` — keep.** Time since the last checkup. Codes 1 through 4 stay in that order, from
within a year to five or more years. Code 8, never, becomes 5 so it continues that order. Codes 7
and 9 become missing.

**34 `BPHIGH4` — drop.** Told that blood pressure is high. Codes 1 yes, 2 pregnancy only, 3 no, 4
borderline, 7 do not know, 9 refused. Used only to build `BPMEDS`: codes 2, 3, and 4 are set to not
medicated before the mode fill. The hypertension fact that stays is `_RFHYPE5`.

**35 `BPMEDS` — keep.** Takes blood pressure medication. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing. If `BPHIGH4` is 2, 3, or 4 the person is not a treated hypertensive,
so this is set to 0 and is not left for the mode fill. `BPHIGH4` is then dropped.

**36 `BLOODCHO` — drop.** Ever had blood cholesterol checked. Codes 7 and 9 are do not know and
refused. The same fact is kept as `_CHOLCHK`, so this copy is not also a column.

**37 `CHOLCHK` — drop.** How long since cholesterol was checked. The codes are time bins, and 7 and
9 are missing. The same fact is kept as `_CHOLCHK`, recoded to checked within five years or not, so
this copy is not also a column.

**38 `TOLDHI2` — drop.** Ever told cholesterol is high. Codes 7 and 9 are missing. The same fact is
kept as `_RFCHOL`, so this copy is not also a column.

**39 `CVDSTRK3` — keep.** Ever told stroke. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9
become missing.

**40 `ASTHMA3` — drop.** Ever told asthma. That is lifetime asthma. Codes 7 and 9 are missing. The
same fact is kept as `_CASTHM1`, which is current asthma, so this copy is not also a column.

**41 `ASTHNOW` — drop.** Still have asthma. Asked only after `ASTHMA3` is yes. Current asthma is
kept as `_CASTHM1`, which already combines the two questions. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**42 `CHCSCNCR` — keep.** Ever told skin cancer. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and
9 become missing.

**43 `CHCOCNCR` — keep.** Ever told another cancer. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing. Kept separately from skin cancer.

**44 `CHCCOPD1` — keep.** Ever told COPD, emphysema, or chronic bronchitis. Code 1 becomes 1 and
code 2 becomes 0. Codes 7 and 9 become missing.

**45 `HAVARTH3` — drop.** Ever told arthritis. Codes 7 and 9 are missing. The same fact is kept as
`_DRDXAR1`, so this copy is not also a column.

**46 `ADDEPEV2` — keep.** Ever told a depressive disorder. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing. The PHQ day counts are kept separately.

**47 `CHCKIDNY` — keep.** Ever told kidney disease. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing.

**48 `DIABETE3` — keep.** Diabetes is not one increasing scale. Three indicators are built, with
code 3, no diabetes, as the reference of all zeros: `diab_yes` for code 1, `diab_pregnancy` for code
2, and `diab_pre` for code 4. Codes 7 and 9 make all three missing. The age product uses `diab_yes`
only.

**49 `DIABAGE2` — keep.** Age when told diabetes. Ages 1 through 97 stay. Codes 98 and 99 become
missing. A blank is everyone who was not asked, not age zero.

**50 `SEX` — keep.** Sex. Code 1, male, stays 1. Code 2, female, becomes 0. The file has no refused
code on this item.

**51 `MARITAL` — keep.** Marital status is a category. The reference is code 1, married. Indicators
are divorced, widowed, separated, never married, and unmarried couple. Code 9 becomes missing on
every indicator.

**52 `EDUCA` — drop.** Education in the raw six-level question. Code 9 is refused. The same fact is
kept as `_EDUCAG`, the ordered four-level calculated education, so this copy is not also a column.

**53 `RENTHOM1` — keep.** Home ownership is a category. The reference is code 1, own. Indicators are
rent and other arrangement. Codes 7 and 9 become missing. This is not the same fact as the income
bracket.

**54 `NUMHHOL2` — drop.** More than one landline in the household. Survey machinery, not a health
measure. Removed before the column score, so the score cannot put it back.

**55 `NUMPHON2` — drop.** Number of residential phones. Asked on a skip, with 7 and 9 as refusals.
Survey machinery, not a health measure. Removed before the column score, so the score cannot put it
back.

**56 `CPDEMO1` — drop.** Cell-phone ownership question used to sort the landline sample. Survey
machinery, not a health measure. Removed before the column score, so the score cannot put it back.

**57 `VETERAN3` — keep.** Veteran. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9 become
missing.

**58 `EMPLOY1` — keep.** Employment is a category, not one linear number. The reference, all zeros,
is code 1, employed for wages. Indicators are self-employed, out of work a year or more, out of work
less than a year, homemaker, student, retired, and unable to work. Code 7 is retired, a real
category, not do not know. Code 9 becomes missing on every indicator. A refusal is then mode filled,
which usually becomes the reference.

**59 `CHILDREN` — drop.** Number of children. Code 88 means none and 99 means refused, so the raw
number is not a count. The same fact is kept as `_CHLDCNT`, recoded to 0 through 5, so this copy is
not also a column.

**60 `INCOME2` — drop.** Household income in the raw brackets. Codes 77 and 99 are do not know and
refused. The same fact is kept as `_INCOMG`, so this copy is not also a column.

**61 `INTERNET` — drop.** Internet use in the past 30 days. Codes 7 and 9 are missing. It is not a
health measure and it is not one of the demographic facts kept with education, income, employment,
and race.

**62 `WEIGHT2` — drop.** Reported weight. The codes mix pounds, a 9000-plus kilogram scale, 7777 do
not know, and 9999 refused. They are not pounds. The same fact is kept as `_BMI5`, so this copy is
not also a column.

**63 `HEIGHT3` — drop.** Reported height. Feet and inches are packed into one integer, and 7777 and
9999 are missing. The same fact is kept as `_BMI5`, so this copy is not also a column.

**64 `PREGNANT` — keep.** Currently pregnant. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9
become missing. Men, sex code 1, are set to 0 here, because the question was not asked of them.
Women who were not asked stay missing for the mode fill.

**65 `QLACTLM2` — keep.** Limited in any activities because of a health problem. Code 1 becomes 1
and code 2 becomes 0. Codes 7 and 9 become missing.

**66 `USEEQUIP` — keep.** Health problem requires special equipment. Code 1 becomes 1 and code 2
becomes 0. Codes 7 and 9 become missing. Distinct from difficulty walking.

**67 `BLIND` — keep.** Blind or serious difficulty seeing. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**68 `DECIDE` — keep.** Serious difficulty concentrating or making decisions. Code 1 becomes 1 and
code 2 becomes 0. Codes 7 and 9 become missing.

**69 `DIFFWALK` — keep.** Serious difficulty walking or climbing stairs. Code 1 becomes 1 and code 2
becomes 0. Codes 7 and 9 become missing.

**70 `DIFFDRES` — keep.** Difficulty dressing or bathing. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**71 `DIFFALON` — keep.** Difficulty doing errands alone. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**72 `SMOKE100` — drop.** Smoked 100 cigarettes in life. Codes 7 and 9 are missing. The same fact is
kept as `_SMOKER3`, so this copy is not also a column.

**73 `SMOKDAY2` — drop.** Smoke every day, some days, or not at all. Asked only after `SMOKE100` is
yes. `_SMOKER3` already combines those answers, and code 9 is refused. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**74 `STOPSMK2` — keep.** Stopped smoking for one day or longer in the past year. Asked of current
smokers. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9 become missing.

**75 `LASTSMK2` — drop.** How long since smoking regularly. Asked of former smokers, and the codes
are not a count of years. `_SMOKER3` keeps former versus never. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**76 `USENOW3` — keep.** Smokeless tobacco, a different product from cigarettes. Every day becomes
2, some days becomes 1, and not at all becomes 0. Codes 7 and 9 become missing.

**77 `ALCDAY5` — drop.** Days drinking alcohol. Codes 101 to 107 are per week, 201 to 230 per month,
888 none, 777 do not know, and 999 refused. Left as one number those are not days. The same fact is
kept as `_RFBING5` and `_RFDRHV5`, so this copy is not also a column.

**78 `AVEDRNK2` — keep.** Average drinks when drinking. Counts 1 through 76 stay. Codes 77 and 99
become missing. Asked only of people who drank.

**79 `DRNK3GE5` — keep.** Binge occasions among drinkers. Counts 1 through 76 stay, and 88, none,
becomes 0. Codes 77 and 99 become missing.

**80 `MAXDRNKS` — keep.** Largest number of drinks on one occasion. Counts 1 through 76 stay. Codes
77 and 99 become missing.

**81 `FRUITJU1` — drop.** Fruit juice frequency. The codes mix per day, per week, per month, 555
never, 777, and 999. The same fact is kept as `_FRTLT1`, so this copy is not also a column.

**82 `FRUIT1` — drop.** Fruit frequency, in the same packed codes as `FRUITJU1`. The same fact is
kept as `_FRTLT1`, so this copy is not also a column.

**83 `FVBEANS` — drop.** Bean frequency, in the packed fruit and vegetable codes. The same fact is
kept as `_VEGLT1`, so this copy is not also a column.

**84 `FVGREEN` — drop.** Dark green vegetable frequency, packed codes. The same fact is kept as
`_VEGLT1`, so this copy is not also a column.

**85 `FVORANG` — drop.** Orange vegetable frequency, packed codes. The same fact is kept as
`_VEGLT1`, so this copy is not also a column.

**86 `VEGETAB1` — drop.** Other vegetable frequency, packed codes. The same fact is kept as
`_VEGLT1`, so this copy is not also a column.

**87 `EXERANY2` — keep.** Any physical activity in the past month. Code 1 becomes 1 and code 2
becomes 0. Codes 7 and 9 become missing. `_TOTINDA` is the dropped copy. The activity-minute
calculations are not kept.

**88 `EXRACT11` — drop.** First activity type. Asked only if `EXERANY2` is yes, and the type codes
are not a dose. `EXERANY2` is the kept activity yes or no. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**89 `EXEROFT1` — drop.** How often the first activity. The codes mix times per day and per week
with 777 and 999. `EXERANY2` is kept. A blank means the question was not asked, not a measured zero.
Mode fill would invent an answer, so the column is removed first.

**90 `EXERHMM1` — drop.** How long the first activity, in hours and minutes packed into one integer,
with 777 and 999 as missing. `EXERANY2` is kept. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**91 `EXRACT21` — drop.** Second activity type. Blank often means there was no second activity.
`EXERANY2` is kept. A blank means the question was not asked, not a measured zero. Mode fill would
invent an answer, so the column is removed first.

**92 `EXEROFT2` — drop.** How often the second activity, with the same packed frequency codes.
`EXERANY2` is kept. A blank means the question was not asked, not a measured zero. Mode fill would
invent an answer, so the column is removed first.

**93 `EXERHMM2` — drop.** How long the second activity, packed hours and minutes. `EXERANY2` is
kept. A blank means the question was not asked, not a measured zero. Mode fill would invent an
answer, so the column is removed first.

**94 `STRENGTH` — drop.** Strength activity frequency, packed like the alcohol day codes, with 888
none and 999 refused. The same fact is kept as `_PASTRNG`, so this copy is not also a column.

**95 `LMTJOIN3` — keep.** Limited by joint symptoms. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing. Most rows were not asked. The missing indicator keeps that fact.

**96 `ARTHDIS2` — keep.** Arthritis limits usual activities. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing. A blank is not a zero.

**97 `ARTHSOCL` — keep.** Arthritis limits social activities. Codes 1 a lot, 2 a little, and 3 not
at all stay in that order. Codes 7 and 9 become missing.

**98 `JOINPAIN` — keep.** Joint pain on a 0 to 10 scale. Those scores stay. Codes 77 and 99 become
missing. A blank means the arthritis pain item was not asked.

**99 `SEATBELT` — drop.** How often a seat belt is used. The raw codes are not a dose, and 7 and 9
are missing. The same fact is kept as `_RFSEAT3`, so this copy is not also a column.

**100 `FLUSHOT6` — keep.** Flu shot in the past year. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing. A blank means the question was not asked.

**101 `FLSHTMY2` — drop.** Month and year of the flu shot, packed into one integer. A blank means
the question was not asked, not a measured zero. Mode fill would invent an answer, so the column is
removed first.

**102 `IMFVPLAC` — drop.** Place the flu shot was given. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**103 `PNEUVAC3` — keep.** Pneumonia vaccination. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and
9 become missing. A blank means the question was not asked.

**104 `HIVTST6` — drop.** Ever tested for HIV. Codes 7 and 9 are missing. The same fact is kept as
`_AIDTST3`, so this copy is not also a column.

**105 `HIVTSTD3` — drop.** Month and year of the last HIV test. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**106 `WHRTST10` — drop.** Where the last HIV test was done. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**107 `PDIABTST` — keep.** Tested for high blood sugar in the past three years. Code 1 becomes 1 and
code 2 becomes 0. Codes 7 and 9 become missing.

**108 `PREDIAB1` — drop.** Told prediabetes in the diabetes module. `diab_pre` is kept. A blank
means the question was not asked, not a measured zero. Mode fill would invent an answer, so the
column is removed first.

**109 `INSULIN` — keep.** Now taking insulin. Asked of people with diabetes. Code 1 becomes 1 and
code 2 becomes 0. Codes 7 and 9 become missing.

**110 `BLDSUGAR` — drop.** How often blood sugar is checked. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**111 `FEETCHK2` — drop.** How often feet are checked for sores. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**112 `DOCTDIAB` — keep.** Times a professional was seen for diabetes. Counts 1 through 76 stay, and
88, none, becomes 0. Codes 77 and 99 become missing.

**113 `CHKHEMO3` — keep.** Times A1C was checked. Counts 1 through 76 stay, and 88, none, becomes 0.
Codes 77, 98, and 99 become missing.

**114 `FEETCHK` — drop.** Times feet were checked by a professional. A blank means the question was
not asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**115 `EYEEXAM` — keep.** Last dilated eye exam among people with diabetes. Codes 1 through 4 stay,
from the past month to two or more years. Code 8, never, becomes 5. Codes 7 and 9 become missing.

**116 `DIABEYE` — keep.** Told diabetes affected the eyes. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**117 `DIABEDU` — keep.** Taken a class on managing diabetes. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**118 `CAREGIV1` — keep.** Provided care to a friend or family member. Code 1 becomes 1 and code 2
becomes 0. Codes 7, 8, and 9 become missing.

**119 `CRGVREL1` — drop.** Relationship to the person cared for. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**120 `CRGVLNG1` — drop.** How long care has been given. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**121 `CRGVHRS1` — drop.** Hours of care per week. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**122 `CRGVPRB1` — drop.** Main health problem of the person cared for. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**123 `CRGVPERS` — drop.** Managed personal care. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**124 `CRGVHOUS` — drop.** Helped with household tasks. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**125 `CRGVMST2` — drop.** Greatest difficulty as a caregiver. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**126 `CRGVEXPT` — drop.** Expected to keep providing care. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**127 `VIDFCLT2` — drop.** Difficulty seeing, even with glasses. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**128 `VIREDIF3` — drop.** Last time vision was checked. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**129 `VIPRFVS2` — drop.** Visited an eye professional. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**130 `VINOCRE2` — drop.** Told to get an eye exam. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**131 `VIEYEXM2` — drop.** Last dilated eye exam. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**132 `VIINSUR2` — drop.** Eye-care insurance. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**133 `VICTRCT4` — drop.** Cataract. A blank means the question was not asked, not a measured zero.
Mode fill would invent an answer, so the column is removed first.

**134 `VIGLUMA2` — drop.** Glaucoma. A blank means the question was not asked, not a measured zero.
Mode fill would invent an answer, so the column is removed first.

**135 `VIMACDG2` — drop.** Macular degeneration. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**136 `CIMEMLOS` — keep.** Confusion or memory loss happening more often. Code 1 becomes 1 and code
2 becomes 0. Codes 7 and 9 become missing.

**137 `CDHOUSE` — drop.** Memory loss interferes with chores. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**138 `CDASSIST` — drop.** Help needed because of memory loss. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**139 `CDHELP` — drop.** Able to get help for memory loss. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**140 `CDSOCIAL` — drop.** Memory loss interferes with activities outside the home. A blank means
the question was not asked, not a measured zero. Mode fill would invent an answer, so the column is
removed first.

**141 `CDDISCUS` — drop.** Discussed memory loss with a professional. A blank means the question was
not asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**142 `WTCHSALT` — keep.** Watching salt intake. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and
9 become missing.

**143 `LONGWTCH` — drop.** How long salt has been watched. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**144 `DRADVISE` — keep.** Doctor advised reducing salt. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**145 `ASTHMAGE` — drop.** Age when first told asthma. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**146 `ASATTACK` — drop.** Asthma attack in the past year. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**147 `ASERVIST` — drop.** Emergency visit for asthma. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**148 `ASDRVIST` — drop.** Urgent doctor visit for asthma. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**149 `ASRCHKUP` — drop.** Routine asthma checkup. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**150 `ASACTLIM` — drop.** Activities limited by asthma. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**151 `ASYMPTOM` — drop.** Asthma symptoms. A blank means the question was not asked, not a measured
zero. Mode fill would invent an answer, so the column is removed first.

**152 `ASNOSLEP` — drop.** Days asthma disturbed sleep. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**153 `ASTHMED3` — drop.** Asthma medication. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**154 `ASINHALR` — drop.** Days an inhaler was used. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**155 `HAREHAB1` — drop.** Outpatient rehab after a heart attack. That is care given because the
person had the event being predicted, so it leaks the label. `CVDINFR4` and `CVDCRHD4` are not in
this file, and `_MICHD` is not rebuilt. A blank means the question was not asked, not a measured
zero. Mode fill would invent an answer, so the column is removed first.

**156 `STREHAB1` — drop.** Outpatient rehab after a stroke. Removed with the rehab items. It is
treatment after a vascular event. The stroke diagnosis stays as `CVDSTRK3`. A blank means the
question was not asked, not a measured zero. Mode fill would invent an answer, so the column is
removed first.

**157 `CVDASPRN` — drop.** Regular aspirin use, from the cardiovascular-health module. Aspirin is
often started after a coronary event, so the answer leaks the label. A blank means the question was
not asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**158 `ASPUNSAF` — drop.** Health problem that makes aspirin unsafe. Same module as `CVDASPRN`, and
a blank means the aspirin questions were not asked. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**159 `RLIVPAIN` — drop.** Limited by joint pain in the arthritis module. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**160 `RDUCHART` — drop.** Arthritis advice about reducing heart risk. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**161 `RDUCSTRK` — drop.** Arthritis advice about reducing stroke risk. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**162 `ARTTODAY` — drop.** Taking medication for arthritis today. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**163 `ARTHWGT` — keep.** Advised to lose weight for arthritis. Code 1 becomes 1 and code 2 becomes
0. Codes 7 and 9 become missing.

**164 `ARTHEXER` — keep.** Advised to exercise for arthritis. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing.

**165 `ARTHEDU` — drop.** Taken a class on arthritis. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**166 `TETANUS` — drop.** Tetanus shot, with a packed Tdap code. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**167 `HPVADVC2` — drop.** Ever received an HPV vaccination. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**168 `HPVADSHT` — drop.** Number of HPV shots. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**169 `SHINGLE2` — keep.** Shingles vaccination. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and
9 become missing.

**170 `HADMAM` — drop.** Ever had a mammogram. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**171 `HOWLONG` — drop.** How long since the last mammogram. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**172 `HADPAP2` — drop.** Ever had a Pap test. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**173 `LASTPAP2` — drop.** How long since the last Pap test. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**174 `HPVTEST` — drop.** Ever had an HPV test. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**175 `HPLSTTST` — drop.** How long since the last HPV test. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**176 `HADHYST2` — drop.** Had a hysterectomy. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**177 `PROFEXAM` — drop.** Clinical breast exam. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**178 `LENGEXAM` — drop.** How long since the clinical breast exam. A blank means the question was
not asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**179 `BLDSTOOL` — keep.** Blood stool test. Code 1 becomes 1 and code 2 becomes 0. Codes 7 and 9
become missing.

**180 `LSTBLDS3` — drop.** How long since the blood stool test. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**181 `HADSIGM3` — keep.** Ever had a sigmoidoscopy. Code 1 becomes 1 and code 2 becomes 0. Codes 7
and 9 become missing.

**182 `HADSGCO1` — drop.** Sigmoidoscopy or colonoscopy. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**183 `LASTSIG3` — drop.** How long since that exam. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**184 `PCPSAAD2` — drop.** Discussed the PSA test advantages. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**185 `PCPSADI1` — drop.** Discussed the PSA test disadvantages. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**186 `PCPSARE1` — drop.** Doctor recommended a PSA test. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**187 `PSATEST1` — drop.** Ever had a PSA test. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**188 `PSATIME` — drop.** How long since the PSA test. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**189 `PCPSARS1` — drop.** PSA result discussed. A blank means the question was not asked, not a
measured zero. Mode fill would invent an answer, so the column is removed first.

**190 `PCPSADE1` — drop.** Decision about the PSA test. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**191 `PCDMDECN` — drop.** Who made the PSA decision. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**192 `SCNTMNY1` — keep.** How often money for housing was a worry. Codes 1 always through 5 never
stay. Codes 7 and 9 become missing.

**193 `SCNTMEL1` — keep.** How often money for meals was a worry. Codes 1 always through 5 never
stay. Codes 7 and 9 become missing.

**194 `SCNTPAID` — drop.** Paid work, social-context module. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**195 `SCNTWRK1` — drop.** Work situation, social-context module. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**196 `SCNTLPAD` — drop.** Behind on rent or mortgage. A blank means the question was not asked, not
a measured zero. Mode fill would invent an answer, so the column is removed first.

**197 `SCNTLWK1` — drop.** Times moved in the past year. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**198 `SXORIENT` — drop.** Sexual orientation, optional module. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**199 `TRNSGNDR` — drop.** Transgender identity, optional module. A blank means the question was not
asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**200 `RCSGENDR` — drop.** Gender of sexual partners, optional module. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**201 `RCSRLTN2` — drop.** Relationship to sexual partners, optional module. A blank means the
question was not asked, not a measured zero. Mode fill would invent an answer, so the column is
removed first.

**202 `CASTHDX2` — drop.** Childhood asthma diagnosis, callback module. A blank means the question
was not asked, not a measured zero. Mode fill would invent an answer, so the column is removed
first.

**203 `CASTHNO2` — drop.** Childhood current asthma, callback module. A blank means the question was
not asked, not a measured zero. Mode fill would invent an answer, so the column is removed first.

**204 `EMTSUPRT` — keep.** How often emotional support is received. Codes 1 always through 5 never
stay in that order. Codes 7 and 9 become missing.

**205 `LSATISFY` — keep.** Life satisfaction. Codes 1 very satisfied through 4 very dissatisfied
stay. Codes 7 and 9 become missing.

**206 `ADPLEASR` — keep.** Little interest or pleasure. Days 1 through 14 stay, 88 becomes 0 days,
and 77 and 99 become missing. The PHQ module is blank for most rows.

**207 `ADDOWN` — keep.** Down or depressed. Days 1 through 14 stay, 88 becomes 0 days, and 77 and 99
become missing. The PHQ module is blank for most rows.

**208 `ADSLEEP` — keep.** Sleep trouble. Days 1 through 14 stay, 88 becomes 0 days, and 77 and 99
become missing. The PHQ module is blank for most rows.

**209 `ADENERGY` — keep.** Tired. Days 1 through 14 stay, 88 becomes 0 days, and 77 and 99 become
missing. The PHQ module is blank for most rows.

**210 `ADEAT1` — keep.** Appetite. Days 1 through 14 stay, 88 becomes 0 days, and 77 and 99 become
missing. The PHQ module is blank for most rows.

**211 `ADFAIL` — keep.** Feeling like a failure. Days 1 through 14 stay, 88 becomes 0 days, and 77
and 99 become missing. The PHQ module is blank for most rows.

**212 `ADTHINK` — keep.** Trouble concentrating. Days 1 through 14 stay, 88 becomes 0 days, and 77
and 99 become missing. The PHQ module is blank for most rows.

**213 `ADMOVE` — keep.** Moving slowly or being fidgety. Days 1 through 14 stay, 88 becomes 0 days,
and 77 and 99 become missing. The PHQ module is blank for most rows.

**214 `MISTMNT` — drop.** Treated badly, optional module. A blank means the question was not asked,
not a measured zero. Mode fill would invent an answer, so the column is removed first.

**215 `ADANXEV` — keep.** Ever told an anxiety disorder. Code 1 becomes 1 and code 2 becomes 0.
Codes 7 and 9 become missing. This optional module is mostly blank.

**216 `QSTVER` — drop.** Questionnaire version number. Survey machinery, not a health measure.
Removed before the column score, so the score cannot put it back.

**217 `QSTLANG` — drop.** Questionnaire language. Survey machinery, not a health measure. Removed
before the column score, so the score cannot put it back.

**218 `MSCODE` — drop.** Metropolitan status code. It is survey geography, not a health measure.
Survey machinery, not a health measure. Removed before the column score, so the score cannot put it
back.

**219 `_STSTR` — drop.** Sample stratum used to build the raking weights. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**220 `_STRWT` — drop.** Stratum weight, a raking input. Survey machinery, not a health measure.
Removed before the column score, so the score cannot put it back.

**221 `_RAWRAKE` — drop.** Raw raking weight. Survey machinery, not a health measure. Removed before
the column score, so the score cannot put it back.

**222 `_WT2RAKE` — drop.** Second raking weight. Survey machinery, not a health measure. Removed
before the column score, so the score cannot put it back.

**223 `_CHISPNC` — drop.** Hispanic origin from the child or sample race block. High missing, and
code 9 is refused. The same fact is kept as `_RACEGR3`, so this copy is not also a column.

**224 `_CRACE1` — drop.** Child or computed race, with 77, 99, and high missing. The same fact is
kept as `_RACEGR3`, so this copy is not also a column.

**225 `_CPRACE` — drop.** Preferred race in the same race block as `_CRACE1`. The same fact is kept
as `_RACEGR3`, so this copy is not also a column.

**226 `_CLLCPWT` — drop.** Combined landline and cell weight used in raking. Survey machinery, not a
health measure. Removed before the column score, so the score cannot put it back.

**227 `_DUALUSE` — drop.** Dual landline and cell use. Code 9 is missing, and the flag adjusts
weights. Survey machinery, not a health measure. Removed before the column score, so the score
cannot put it back.

**228 `_DUALCOR` — drop.** Dual-frame correction factor applied to the sampling weights. Survey
machinery, not a health measure. Removed before the column score, so the score cannot put it back.

**229 `_LLCPWT` — drop.** Final raking weight `_LLCPWT`. It is the survey weight, not body weight.
Survey machinery, not a health measure. Removed before the column score, so the score cannot put it
back.

**230 `_RFHLTH` — drop.** Calculated fair or poor health from `GENHLTH`. Code 9 is missing. The same
fact is kept as `GENHLTH`, kept as 1 excellent through 5 poor, so this copy is not also a column.

**231 `_HCVU651` — drop.** Health coverage for ages 18 to 64. Code 9 is mostly age 65 and older, not
a refusal, so treating 9 as missing would drop older adults. The same fact is kept as `HLTHPLN1`, so
this copy is not also a column.

**232 `_RFHYPE5` — keep.** Calculated high blood pressure. CDC uses 2 for yes and 1 for no. Yes
becomes 1, no becomes 0, and 9 becomes missing. Pregnancy-only and borderline answers sit in that 9.

**233 `_CHOLCHK` — keep.** Cholesterol check. Code 1, within five years, becomes 1. Codes 2 and 3,
older check or never, become 0. Code 9 becomes missing.

**234 `_RFCHOL` — keep.** Calculated high cholesterol. CDC uses 2 for yes and 1 for no. Yes becomes
1, no becomes 0, and 9 becomes missing. `TOLDHI2` is the dropped copy. `rfchol_missing` is 1 on
those missing rows, so the mode fill does not hide the refusal.

**235 `_LTASTH1` — drop.** Calculated lifetime asthma. Code 9 is missing. The same fact is kept as
`_CASTHM1`, current asthma, so this copy is not also a column.

**236 `_CASTHM1` — keep.** Current asthma. This calculated item uses 2 for yes and 1 for no. Yes
becomes 1, no becomes 0, and 9 becomes missing. Lifetime asthma `ASTHMA3` is the dropped copy.

**237 `_ASTHMS1` — drop.** Four-level calculated asthma status. The numbers are categories, and code
9 is missing. The same fact is kept as `_CASTHM1`, so this copy is not also a column.

**238 `_DRDXAR1` — keep.** Doctor-diagnosed arthritis. Code 1 becomes 1 and code 2 becomes 0.
`HAVARTH3` is the dropped copy.

**239 `_PRACE1` — drop.** Preferred race, with code 99 refused. The same fact is kept as `_RACEGR3`,
so this copy is not also a column.

**240 `_MRACE1` — drop.** Multiracial preferred race, with code 99 refused. The same fact is kept as
`_RACEGR3`, so this copy is not also a column.

**241 `_HISPANC` — drop.** Hispanic origin. Code 9 is refused. The same fact is kept as `_RACEGR3`,
whose Hispanic level is its own indicator, so this copy is not also a column.

**242 `_RACE` — drop.** Detailed race and ethnicity. The codes are categories, not a scale. The same
fact is kept as `_RACEGR3`, so this copy is not also a column.

**243 `_RACEG21` — drop.** White versus all other groups. A coarser copy of race. The same fact is
kept as `_RACEGR3`, so this copy is not also a column.

**244 `_RACEGR3` — keep.** Race and ethnicity are categories. The reference is code 1, white
non-Hispanic. Indicators are black non-Hispanic, other race non-Hispanic, multiracial non-Hispanic,
and Hispanic. Code 9 becomes missing on every indicator. The other race columns are dropped copies.

**245 `_RACE_G1` — drop.** Five-level race with refusals already dropped. Same fact as `_RACEGR3`.
The same fact is kept as `_RACEGR3`, so this copy is not also a column.

**246 `_AGEG5YR` — drop.** Fourteen age bands. Code 9 is a real band in this file, not a refusal, so
it must not be read as missing or as a dose. The same fact is kept as `_AGE80`, so this copy is not
also a column.

**247 `_AGE65YR` — drop.** Under 65 versus 65 and older. Code 3 is the missing age group. The same
fact is kept as `_AGE80`, so this copy is not also a column.

**248 `_AGE80` — keep.** Age in years. Values 18 through 79 stay as years, and 80 means 80 or older.
The square of this column is added later. `_AGEG5YR`, `_AGE65YR`, and `_AGE_G` are the dropped
copies.

**249 `_AGE_G` — drop.** Six age groups. A coarser copy of age in years. The same fact is kept as
`_AGE80`, so this copy is not also a column.

**250 `HTIN4` — drop.** Height in inches, calculated from `HEIGHT3`. The same fact is kept as
`_BMI5`, so this copy is not also a column.

**251 `HTM4` — drop.** Height in meters, calculated from `HEIGHT3`. The same fact is kept as
`_BMI5`, so this copy is not also a column.

**252 `WTKG3` — drop.** Weight in kilograms, calculated from `WEIGHT2`. The same fact is kept as
`_BMI5`, so this copy is not also a column.

**253 `_BMI5` — keep.** Body mass index. In this file the values are already body mass units, such
as 26.6, not the CDC integer times 100. Blanks are the missing values. There is no 9999 sentinel
here, and a large value such as 77 is a BMI, not a do-not-know code. Values above 60 are set to 60
before the square, so a few extreme rows do not dominate `bmi5_sq`. `WEIGHT2` and `HEIGHT3` are the
dropped copies.

**254 `_BMI5CAT` — drop.** Four BMI categories of `_BMI5`. The same fact is kept as `_BMI5`, the
continuous body mass value, so this copy is not also a column.

**255 `_RFBMI5` — drop.** Overweight or obese flag. Code 9 is a missing BMI, not a third body size.
The same fact is kept as `_BMI5`, so this copy is not also a column.

**256 `_CHLDCNT` — keep.** Number of children. CDC code 1 means none, so the codes are shifted to 0
through 5, with 5 meaning five or more. Code 9 becomes missing. `CHILDREN` is the dropped copy,
because 88 there means none.

**257 `_EDUCAG` — keep.** Calculated education, kept as the ordered codes 1 through 4. Higher means
more education. Code 9 becomes missing. `EDUCA` is the dropped copy.

**258 `_INCOMG` — keep.** Calculated income, kept as the ordered codes 1 through 5. Higher means
more income. Code 9 becomes missing. `INCOME2` is the dropped copy. `incomg_missing` is 1 on those
rows. The mode alone would mark a refusal as the top income bracket.

**259 `_SMOKER3` — keep.** Calculated smoking. Every day, code 1, becomes 3. Some days, code 2,
becomes 2. Former, code 3, becomes 1. Never, code 4, becomes 0. Code 9 becomes missing. `SMOKE100`
and `SMOKDAY2` are the dropped copies.

**260 `_RFSMOK3` — drop.** Current smoker flag. Code 9 is missing smoking. The same fact is kept as
`_SMOKER3`, kept as never 0, former 1, some days 2, every day 3, so this copy is not also a column.

**261 `DRNKANY5` — drop.** Any alcohol in the past 30 days. Codes 7 and 9 are missing. The same fact
is kept as `_RFBING5` and `_RFDRHV5`, which already mark non-drinkers as no, so this copy is not
also a column.

**262 `DROCDY3_` — drop.** Calculated drink occasions, with 900 as a missing sentinel. The same fact
is kept as `_RFBING5` and `_RFDRHV5`, so this copy is not also a column.

**263 `_RFBING5` — keep.** Calculated binge drinking. CDC uses 2 for yes and 1 for no. Yes becomes
1, no becomes 0, and 9 becomes missing.

**264 `_DRNKWEK` — drop.** Calculated drinks per week, with 99900 as a missing sentinel. The same
fact is kept as `_RFBING5` and `_RFDRHV5`, so this copy is not also a column.

**265 `_RFDRHV5` — keep.** Calculated heavy drinking, which is not the same fact as binge drinking.
CDC uses 2 for yes and 1 for no. Yes becomes 1, no becomes 0, and 9 becomes missing.

**266 `FTJUDA1_` — drop.** Fruit juice times per day. The same fact is kept as `_FRTLT1` and
`_VEGLT1`, so this copy is not also a column.

**267 `FRUTDA1_` — drop.** Fruit times per day. The same fact is kept as `_FRTLT1` and `_VEGLT1`, so
this copy is not also a column.

**268 `BEANDAY_` — drop.** Bean times per day. The same fact is kept as `_FRTLT1` and `_VEGLT1`, so
this copy is not also a column.

**269 `GRENDAY_` — drop.** Green vegetable times per day. The same fact is kept as `_FRTLT1` and
`_VEGLT1`, so this copy is not also a column.

**270 `ORNGDAY_` — drop.** Orange vegetable times per day. The same fact is kept as `_FRTLT1` and
`_VEGLT1`, so this copy is not also a column.

**271 `VEGEDA1_` — drop.** Other vegetable times per day. The same fact is kept as `_FRTLT1` and
`_VEGLT1`, so this copy is not also a column.

**272 `_MISFRTN` — drop.** Count of missing fruit items. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**273 `_MISVEGN` — drop.** Count of missing vegetable items. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**274 `_FRTRESP` — drop.** Answered the fruit questions. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**275 `_VEGRESP` — drop.** Answered the vegetable questions. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**276 `_FRUTSUM` — drop.** Sum of the fruit frequencies. Values can still sit on a 99 hole. The same
fact is kept as `_FRTLT1` and `_VEGLT1`, so this copy is not also a column.

**277 `_VEGESUM` — drop.** Sum of the vegetable frequencies. The same fact is kept as `_FRTLT1` and
`_VEGLT1`, so this copy is not also a column.

**278 `_FRTLT1` — keep.** Fruit less than once a day. Code 1 becomes 1 and code 2 becomes 0. Code 9
becomes missing. The raw frequency items are dropped because their packed codes are not counts.
`frtlt1_missing` marks that hole before the mode fill.

**279 `_VEGLT1` — keep.** Vegetables less than once a day. Code 1 becomes 1 and code 2 becomes 0.
Code 9 becomes missing. `veglt1_missing` marks that hole before the mode fill.

**280 `_FRT16` — drop.** Fruit response bookkeeping flag, nearly constant. Bookkeeping for the fruit
and vegetable calculation, not a diet measure.

**281 `_VEG23` — drop.** Vegetable response bookkeeping flag, nearly constant. Bookkeeping for the
fruit and vegetable calculation, not a diet measure.

**282 `_FRUITEX` — drop.** Fruit exclusion flag. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**283 `_VEGETEX` — drop.** Vegetable exclusion flag. Bookkeeping for the fruit and vegetable
calculation, not a diet measure.

**284 `_TOTINDA` — drop.** Calculated any leisure activity. Code 9 folds nonresponse into one bin.
The same fact is kept as `EXERANY2`, so this copy is not also a column.

**285 `METVL11_` — drop.** MET value of the first activity. Derived from the same activity answers.
The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this copy is
not also a column.

**286 `METVL21_` — drop.** MET value of the second activity. Derived from the same activity answers.
The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this copy is
not also a column.

**287 `MAXVO2_` — drop.** Estimated maximum oxygen uptake. Code 999 is missing, and it is not a
measured VO2. Derived from the same activity answers. The same fact is kept as `EXERANY2` for any
activity and `_PASTRNG` for strength, so this copy is not also a column.

**288 `FC60_` — drop.** Estimated functional capacity. Code 999 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**289 `ACTIN11_` — drop.** Aerobic flag for the first activity. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**290 `ACTIN21_` — drop.** Aerobic flag for the second activity. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**291 `PADUR1_` — drop.** Minutes of the first activity. Derived from the same activity answers. The
same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this copy is not
also a column.

**292 `PADUR2_` — drop.** Minutes of the second activity. Derived from the same activity answers.
The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this copy is
not also a column.

**293 `PAFREQ1_` — drop.** Times per week of the first activity. Code 99 is missing. Derived from
the same activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for
strength, so this copy is not also a column.

**294 `PAFREQ2_` — drop.** Times per week of the second activity. Code 99 is missing. Derived from
the same activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for
strength, so this copy is not also a column.

**295 `_MINAC11` — drop.** Minutes per week of the first activity. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**296 `_MINAC21` — drop.** Minutes per week of the second activity. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**297 `STRFREQ_` — drop.** Strength sessions per week. Code 99 is missing. The same fact is kept as
`_PASTRNG`, so this copy is not also a column.

**298 `PAMISS1_` — drop.** Flag that physical-activity answers are missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**299 `PAMIN11_` — drop.** Minutes of the first activity after a missing rule. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**300 `PAMIN21_` — drop.** Minutes of the second activity after a missing rule. Derived from the
same activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for
strength, so this copy is not also a column.

**301 `PA1MIN_` — drop.** Total activity minutes. Derived from the same activity answers. The same
fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this copy is not also a
column.

**302 `PAVIG11_` — drop.** Vigorous-equivalent minutes of the first activity. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**303 `PAVIG21_` — drop.** Vigorous-equivalent minutes of the second activity. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**304 `PA1VIGM_` — drop.** Total vigorous-equivalent minutes. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**305 `_PACAT1` — drop.** Four-level activity category. Code 9 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**306 `_PAINDX1` — drop.** Meets the aerobic guideline. Code 9 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**307 `_PA150R2` — drop.** 150-minute activity category. Code 9 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**308 `_PA300R2` — drop.** 300-minute activity category. Code 9 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**309 `_PA30021` — drop.** 300-minute yes or no. Code 9 is missing. Derived from the same activity
answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength, so this
copy is not also a column.

**310 `_PASTRNG` — keep.** Muscle strengthening two or more times a week. Code 1 becomes 1 and code
2 becomes 0. Code 9 becomes missing. The raw `STRENGTH` frequency is the dropped copy.
`pastrng_missing` marks that hole before the mode fill.

**311 `_PAREC1` — drop.** Activity recommendation category. Code 9 is missing. Derived from the same
activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for strength,
so this copy is not also a column.

**312 `_PASTAE1` — drop.** Meets the aerobic and strength guideline. Code 9 is missing. Derived from
the same activity answers. The same fact is kept as `EXERANY2` for any activity and `_PASTRNG` for
strength, so this copy is not also a column.

**313 `_LMTACT1` — drop.** Calculated limitation of usual activities by arthritis. Code 3 means no
arthritis, which is the majority, so a linear reading treats no arthritis as a high score. The
diagnosis is kept as `_DRDXAR1`. Limitation for the whole sample is kept as `DIFFWALK` and
`QLACTLM2`.

**314 `_LMTWRK1` — drop.** Calculated work limitation from arthritis, with the same non-linear code
3 for no arthritis. `_DRDXAR1`, `DIFFWALK`, and `QLACTLM2` are kept.

**315 `_LMTSCL1` — drop.** Calculated social limitation from arthritis. The codes are categories,
and code 4 is no arthritis. `_DRDXAR1` is kept.

**316 `_RFSEAT2` — drop.** Usually wears a seat belt, an alternate calculated cut. Code 9 is
missing. The same fact is kept as `_RFSEAT3`, so this copy is not also a column.

**317 `_RFSEAT3` — keep.** Always or nearly always wears a seat belt. Code 1 becomes 1 and code 2
becomes 0. Code 9 becomes missing. `rfseat3_missing` marks that hole before the mode fill.

**318 `_FLSHOT6` — keep.** Calculated flu shot. Code 1, yes, stays 1. Code 2, no, becomes 0. Code 9
becomes missing. Younger adults are often unscored, and that hole is a missing indicator rather than
a dropped column.

**319 `_PNEUMO2` — keep.** Calculated pneumonia vaccination. Code 1 stays 1, code 2 becomes 0, and
code 9 becomes missing. Same age hole as `_FLSHOT6`.

**320 `_AIDTST3` — keep.** Ever tested for HIV. Code 1 becomes 1 and code 2 becomes 0. Code 9
becomes missing. `HIVTST6` is the dropped copy. `aidtst3_missing` marks that hole before the mode
fill.

## Nonlinear terms

These are not original columns. They are added after standardization of the codebook matrix, then
standardized with it.

`age80_sq` is the square of `_AGE80`. `bmi5_sq` is the square of capped `_BMI5`. `age_x_sex`,
`age_x_smoker`, and `age_x_diab_yes` are `_AGE80` times `SEX`, `_SMOKER3`, and `diab_yes`.
`age_x_genhlth` and `age_x_rfchol` are `_AGE80` times `GENHLTH` and `_RFCHOL`. `bmi_x_smoker` is
`_BMI5` times `_SMOKER3`. Pregnancy-only diabetes and prediabetes stay linear. Logistic regression
receives these same columns. The sigmoid bends the score. It does not create these products.

## Model column order

The codebook matrix has 115 columns, in this order. Squares and products are added after
standardization and are not in this list.

0. `GENHLTH`
1. `PHYSHLTH`
2. `MENTHLTH`
3. `HLTHPLN1`
4. `PERSDOC2`
5. `MEDCOST`
6. `CHECKUP1`
7. `BPMEDS`
8. `_RFHYPE5`
9. `_CHOLCHK`
10. `_RFCHOL`
11. `CVDSTRK3`
12. `_CASTHM1`
13. `CHCSCNCR`
14. `CHCOCNCR`
15. `CHCCOPD1`
16. `_DRDXAR1`
17. `ADDEPEV2`
18. `CHCKIDNY`
19. `diab_yes`
20. `diab_pregnancy`
21. `diab_pre`
22. `QLACTLM2`
23. `USEEQUIP`
24. `BLIND`
25. `DECIDE`
26. `DIFFWALK`
27. `DIFFDRES`
28. `DIFFALON`
29. `_SMOKER3`
30. `USENOW3`
31. `EXERANY2`
32. `_PASTRNG`
33. `_RFBING5`
34. `_RFDRHV5`
35. `_FRTLT1`
36. `_VEGLT1`
37. `_RFSEAT3`
38. `_AIDTST3`
39. `SEX`
40. `_AGE80`
41. `_BMI5`
42. `_EDUCAG`
43. `_INCOMG`
44. `_CHLDCNT`
45. `VETERAN3`
46. `employ_self`
47. `employ_out_long`
48. `employ_out_short`
49. `employ_homemaker`
50. `employ_student`
51. `employ_retired`
52. `employ_unable`
53. `marital_divorced`
54. `marital_widowed`
55. `marital_separated`
56. `marital_never`
57. `marital_couple`
58. `housing_rent`
59. `housing_other`
60. `race_black`
61. `race_other`
62. `race_multi`
63. `race_hispanic`
64. `POORHLTH`
65. `LMTJOIN3`
66. `ARTHDIS2`
67. `ARTHSOCL`
68. `JOINPAIN`
69. `ARTHWGT`
70. `ARTHEXER`
71. `CIMEMLOS`
72. `PDIABTST`
73. `INSULIN`
74. `DIABEYE`
75. `DIABEDU`
76. `DIABAGE2`
77. `CHKHEMO3`
78. `DOCTDIAB`
79. `EYEEXAM`
80. `WTCHSALT`
81. `DRADVISE`
82. `AVEDRNK2`
83. `DRNK3GE5`
84. `MAXDRNKS`
85. `STOPSMK2`
86. `FLUSHOT6`
87. `PNEUVAC3`
88. `_FLSHOT6`
89. `_PNEUMO2`
90. `SHINGLE2`
91. `BLDSTOOL`
92. `HADSIGM3`
93. `ADANXEV`
94. `CAREGIV1`
95. `PREGNANT`
96. `EMTSUPRT`
97. `LSATISFY`
98. `SCNTMNY1`
99. `SCNTMEL1`
100. `ADPLEASR`
101. `ADDOWN`
102. `ADSLEEP`
103. `ADENERGY`
104. `ADEAT1`
105. `ADFAIL`
106. `ADTHINK`
107. `ADMOVE`
108. `incomg_missing`
109. `rfchol_missing`
110. `frtlt1_missing`
111. `veglt1_missing`
112. `rfseat3_missing`
113. `pastrng_missing`
114. `aidtst3_missing`
