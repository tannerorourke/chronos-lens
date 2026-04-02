

MODEL_VECS          = ["z_enc", "z_pred", "z_target"]
MODEL_PRED_VECS     = ["z_pred", "z_target", "pred_error", "z_enc_pooled"]
SUPERVISED_VECTORS  = ["z_enc_pooled"]


ALL_TASKS = ["readmit_30d", "escalation", "icd_block", "escalation_type"]
LABEL_ICD10_PREFIX = "F"

# =============================================================================
# Drug class definitions
# 
# Fefines drug classes and their associated medications. Medications have been
# cross-referenced with MIMIC prescriptions data and grouped based on 
# FDA classification.

DRUG_CLASSES = {
    # =========================================================================
    # OVERDOSE REVERSAL - singular administration (atomic event)
    "reversal": [
        "naloxone",
        "flumazenil",
    ],
    "ssri": [
        "sertraline",         # Zoloft
        "escitalopram",       # Lexapro
        "citalopram",         # Celexa
        "fluoxetine",         # Prozac
        "paroxetine",         # Paxil
        "fluvoxamine",        # Luvox, used for OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : SNRIs
    "snri": [
        "duloxetine",         # Cymbalta
        "venlafaxine",        # Effexor
        "desvenlafaxine",     # Pristiq, active metabolite of venlafaxine
        "milnacipran",        # Savella, fibromyalgia
        "levomilnacipran",    # Fetzima
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : Other mechanisms
    "antidepressant_other": [
        "bupropion",          # Wellbutrin
        "mirtazapine",        # Remeron
        "trazodone",          # Desyrel
        "nefazodone",         # Serzone
        "vilazodone",         # Viibryd
        "vortioxetine",       # Trintellix
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : Tricyclics (TCAs), Common for pain, depression, neuropathy
    "tca": [
        "amitriptyline",      # Elavil
        "nortriptyline",      # Pamelor
        "desipramine",        # Norpramin
        "imipramine",         # Tofranil
        "doxepin",            # Sinequan - also used for sleep/itching
        "clomipramine",       # Anafranil - OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : MAOIs (rare but present)
    "maoi": [
        "phenelzine",         # Nardil
        "tranylcypromine",    # Parnate
        "selegiline",         # Emsam / Eldepryl, also used for Parkinsons
    ],
    # =========================================================================
    # ANXIOLYTICS : Non-Benzodiazepine
    "anxiolytic": [
        "buspirone",          # Buspar
        "hydroxyzine",        # Vistaril/Atarax, very common for anxiety/itch
        "pregabalin",         # Lyrica, anxiety, nerve pain, fibromyalgia
        "gabapentin",         # Neurontin, off-label anxiety
    ],
    # =========================================================================
    # BENZODIAZEPINES
    "benzodiazepine": [
        "diazepam",           # Valium
        "clonazepam",         # Klonopin
        "lorazepam",          # Ativan
        "midazolam",          # Versed, ICU sedation
        "alprazolam",         # Xanax
        "chlordiazepoxide",   # Librium, alcohol withdrawal
        "oxazepam",           # Serax
        "temazepam",          # Restoril, sleep
        "triazolam",          # Halcion
        "clorazepate",        # Tranxene
    ],
    # =========================================================================
    # ANTIPSYCHOTICS : 1st Generation (Typical)
    "antipsychotic_typical": [
        "haloperidol",        # Haldol - extremely common in MIMIC (agitation, delirium)
        "chlorpromazine",     # Thorazine
        "fluphenazine",       # Prolixin
        "perphenazine",       # Trilafon
        "thiothixene",        # Navane
        "loxapine",           # Loxitane
        "pimozide",           # Orap - rare
        "prochlorperazine",   # Compazine - often used as antiemetic
    ],
    # =========================================================================
    # ANTIPSYCHOTICS : 2nd Generation (Atypical)
    "antipsychotic_atypical": [
        "quetiapine",         # Seroquel, most common (psychosis, sleep, bipolar)
        "olanzapine",         # Zyprexa
        "risperidone",        # Risperdal
        "aripiprazole",       # Abilify
        "ziprasidone",        # Geodon
        "clozapine",          # Clozaril, treatment-resistant schizophrenia
        "paliperidone",       # Invega
        "lurasidone",         # Latuda
        "brexpiprazole",      # Rexulti
        "cariprazine",        # Vraylar
        "asenapine",          # Saphris
    ],
    # =========================================================================
    # MOOD STABILIZERS / ANTICONVULSANTS
    "mood_stabilizer": [
        "lithium",            # Lithobid, Eskalith, very common
        "valproic acid",      # Depakote/Depakene, very common
        "valproate",          # alternate naming in prescriptions
        "divalproex",         # Depakote
        "carbamazepine",      # Tegretol
        "oxcarbazepine",      # Trileptal
        "lamotrigine",        # Lamictal
        "topiramate",         # Topamax, off-label mood, migraine
    ],
    # =========================================================================
    # ADHD
    "adhd": [
        "methylphenidate",    # Ritalin, Concerta
        "dexmethylphenidate", # Focalin
        "amphetamine",        # Adderall (mixed amphetamine salts)
        "dextroamphetamine",  # Dexedrine
        "lisdexamfetamine",   # Vyvanse
        "atomoxetine",        # Strattera, non-stimulant
        "guanfacine",         # Intuniv, non-stimulant
        "clonidine",          # Kapvay, also used for ADHD, very common
    ],
    # =========================================================================
    "cognitive": [
        "donepezil",          # Aricept
        "memantine",          # Namenda
        "rivastigmine",       # Exelon
        "galantamine",        # Razadyne
    ],
    # =========================================================================
    # MEDICATION ASSISTED TREATMENT (Alcohol, Opioid Dependence)
    "mat": [
        "buprenorphine",      # Suboxone (w/ naloxone), Subutex
        "methadone",          # Methadone
        "naltrexone",         # Vivitrol, ReVia
    ],
    # =========================================================================
    "smoking_cessation": [
        "nicotine",
        "varenicline",        # Chantix
    ],
    # =========================================================================
    # (pain / SUD risk tracking)
    "opioid": [               
        "hydrocodone",        # Vicodin, Norco
        "oxycodone",          # OxyContin, Percocet
        "codeine",            # Tylenol #3
        "fentanyl",           # Duragesic patch, IV (very common in ICU)
        "tramadol",           # Ultram
        "morphine",           # MS Contin, IV, extremely common
        "hydromorphone",      # Dilaudid, very common
        "meperidine",         # Demerol
        "methadone",          # NOTE: also in MAT; context determines category
        "alfentanil",         # Anesthesia
        "sufentanil",         # Anesthesia
        "remifentanil",       # Anesthesia
        "tapentadol",         # Nucynta
    ],
    # =========================================================================
    "sleep": [
        "zolpidem",           # Ambien
        "eszopiclone",        # Lunesta
        "suvorexant",         # Belsomra
        "ramelteon",          # Rozerem
        "melatonin",          # OTC but frequently appears
        # trazodone also used for sleep - listed under antidepressant_other
        # quetiapine low-dose for sleep - listed under antipsychotic_atypical
    ],
}
# =============================================================================

# =============================================================================
# Psychiatric drug class names
PSYCH_CLASSES: frozenset[str] = frozenset({
    "reversal", "ssri", "snri", "antidepressant_other", "tca", "maoi",
    "anxiolytic", "benzodiazepine",
    "antipsychotic_typical", "antipsychotic_atypical",
    "mood_stabilizer", "adhd", "mat", "smoking_cessation", "opioid", "sleep"
})

# Flat set of all psychiatric medication names (lowercased)
PSYCH_MEDS_FLAT: set[str] = set()
for _cls in PSYCH_CLASSES:
    if _cls in DRUG_CLASSES:
        PSYCH_MEDS_FLAT.update(m.lower() for m in DRUG_CLASSES[_cls])
# =============================================================================

ESCALATION_CRITERIA = [
    "new_subcategory", "severity_increase", "new_specifier",
    "f32_to_f33", "med_initiation", "new_drug_class",
]

# =============================================================================
# ICD-10 F-Code Dictionary, by severity
#
# Structure:
# - Category blocks (e.g. "F30-F39") -> nested dict of subcategories/codes
# - Subcategories (e.g. "F32") -> nested dict of leaf codes
# - Leaf codes (e.g. "F32.0") -> int
#    -1  : code exists but carries no severity information
#    0   : sentinel for "unspecified" / "other" within a severity family
#    1-N : ordinal severity (1=mild, 2=moderate, 3=severe, 4=severe+psychotic/extreme)
#
# Severity is assigned when ICD-10 explicitly uses the subcategory digit to
# encode clinical severity (mild / moderate / severe / with psychotic features).
# Codes that differ only in episode type, specifier, or aetiology get either 0 or -1.
ICD10_F_SEVERITY = {
    # ------------------------------------------------------------------ #
    # F01-F09  Organic, including symptomatic, mental disorders           #
    # ------------------------------------------------------------------ #
    "F01-F09": {

        "F01": {  # Vascular dementia
            "F01.50": 0,   # unspecified severity
            "F01.51": -1,  # with behavioural disturbance (specifier, not severity)
            "F01.52": -1,  # with combative behaviour
            "F01.53": -1,  # with psychotic disturbance
            "F01.54": -1,  # with anxiety
            "F01.A":  -1,  # mild neurocognitive impairment
            "F01.B":  -1,  # moderate neurocognitive impairment
            "F01.C":  -1,  # severe neurocognitive impairment
        },

        "F02": {  # Dementia in other diseases classified elsewhere
            "F02.80": 0,
            "F02.81": -1,
            "F02.82": -1,
            "F02.83": -1,
            "F02.84": -1,
            "F02.A":  -1,
            "F02.B":  -1,
            "F02.C":  -1,
        },

        "F03": {  # Unspecified dementia
            "F03.90": 0,
            "F03.91": -1,
            "F03.92": -1,
            "F03.93": -1,
            "F03.94": -1,
            "F03.A":  -1,
            "F03.B":  -1,
            "F03.C":  -1,
        },

        "F04":  -1,  # Amnestic disorder due to known physiological condition
        "F05":  -1,  # Delirium due to known physiological condition

        "F06": {  # Other mental disorders due to known physiological condition
            "F06.0":  -1,
            "F06.1":  -1,
            "F06.2":  -1,
            "F06.30": 0,
            "F06.31": -1,
            "F06.32": -1,
            "F06.33": -1,
            "F06.34": -1,
            "F06.4":  -1,
            "F06.8":  -1,
        },

        "F07": {  # Personality and behavioural disorders due to known physiological condition
            "F07.0":  -1,
            "F07.81": -1,
            "F07.89": -1,
            "F07.9":  0,
        },

        "F09": -1,  # Unspecified mental disorder due to known physiological condition
    },

    # ------------------------------------------------------------------ #
    # F10-F19  Mental and behavioural disorders due to psychoactive       #
    #          substance use                                              #
    # ------------------------------------------------------------------ #
    "F10-F19": {
        # Pattern is identical across all substances F10-F19:
        #   .10/.20 = uncomplicated use/dependence        -> -1
        #   .121/.221 = with intoxication delirium        -> -1
        #   .14/.24  = with induced mood disorder         -> -1
        #   etc.
        # None of the substance-use subcategories encode a severity scale
        # (mild/moderate/severe refer to USE DISORDER level, which IS a
        # severity scale in DSM-5 but ICD-10-CM encodes it differently -
        # F1x.10=unspecified, no explicit 1/2/3 digit - so -1 throughout).

        "F10": {  # Alcohol
            "F10.10": 0,  "F10.11": -1, "F10.120": -1, "F10.121": -1,
            "F10.129": 0, "F10.13": -1, "F10.14": -1,  "F10.150": -1,
            "F10.151": -1,"F10.159": 0, "F10.180": -1, "F10.181": -1,
            "F10.182": -1,"F10.188": -1,"F10.19": 0,
            "F10.20": 0,  "F10.21": -1, "F10.220": -1, "F10.221": -1,
            "F10.229": 0, "F10.23": -1, "F10.24": -1,  "F10.250": -1,
            "F10.251": -1,"F10.259": 0, "F10.26": -1,  "F10.27": -1,
            "F10.280": -1,"F10.281": -1,"F10.282": -1, "F10.288": -1,
            "F10.29": 0,
            "F10.920": -1,"F10.921": -1,"F10.929": 0,  "F10.93": -1,
            "F10.94": -1, "F10.950": -1,"F10.951": -1, "F10.959": 0,
            "F10.96": -1, "F10.97": -1, "F10.980": -1, "F10.981": -1,
            "F10.982": -1,"F10.988": -1,"F10.99": 0,
        },

        "F11": {  # Opioid
            "F11.10": 0,  "F11.11": -1, "F11.120": -1, "F11.121": -1,
            "F11.122": -1,"F11.129": 0, "F11.13": -1,  "F11.14": -1,
            "F11.150": -1,"F11.151": -1,"F11.159": 0,  "F11.181": -1,
            "F11.182": -1,"F11.188": -1,"F11.19": 0,
            "F11.20": 0,  "F11.21": -1, "F11.220": -1, "F11.221": -1,
            "F11.222": -1,"F11.229": 0, "F11.23": -1,  "F11.24": -1,
            "F11.250": -1,"F11.251": -1,"F11.259": 0,  "F11.281": -1,
            "F11.282": -1,"F11.288": -1,"F11.29": 0,
            "F11.90": 0,  "F11.920": -1,"F11.921": -1, "F11.922": -1,
            "F11.929": 0, "F11.93": -1, "F11.94": -1,  "F11.950": -1,
            "F11.951": -1,"F11.959": 0, "F11.981": -1, "F11.982": -1,
            "F11.988": -1,"F11.99": 0,
        },

        "F12": {  # Cannabis
            "F12.10": 0,  "F12.11": -1, "F12.120": -1, "F12.121": -1,
            "F12.122": -1,"F12.129": 0, "F12.13": -1,  "F12.150": -1,
            "F12.151": -1,"F12.159": 0, "F12.180": -1, "F12.188": -1,
            "F12.19": 0,
            "F12.20": 0,  "F12.21": -1, "F12.220": -1, "F12.221": -1,
            "F12.222": -1,"F12.229": 0, "F12.23": -1,  "F12.250": -1,
            "F12.251": -1,"F12.259": 0, "F12.280": -1, "F12.288": -1,
            "F12.29": 0,
            "F12.90": 0,  "F12.920": -1,"F12.921": -1, "F12.922": -1,
            "F12.929": 0, "F12.93": -1, "F12.950": -1, "F12.951": -1,
            "F12.959": 0, "F12.980": -1,"F12.988": -1, "F12.99": 0,
        },

        "F13": {  # Sedative/hypnotic/anxiolytic
            "F13.10": 0,  "F13.11": -1, "F13.120": -1, "F13.121": -1,
            "F13.129": 0, "F13.13": -1, "F13.14": -1,  "F13.150": -1,
            "F13.151": -1,"F13.159": 0, "F13.180": -1, "F13.181": -1,
            "F13.182": -1,"F13.188": -1,"F13.19": 0,
            "F13.20": 0,  "F13.21": -1, "F13.220": -1, "F13.221": -1,
            "F13.229": 0, "F13.23": -1, "F13.24": -1,  "F13.250": -1,
            "F13.251": -1,"F13.259": 0, "F13.26": -1,  "F13.27": -1,
            "F13.280": -1,"F13.281": -1,"F13.282": -1, "F13.288": -1,
            "F13.29": 0,
            "F13.90": 0,  "F13.920": -1,"F13.921": -1, "F13.929": 0,
            "F13.93": -1, "F13.94": -1, "F13.950": -1, "F13.951": -1,
            "F13.959": 0, "F13.96": -1, "F13.97": -1,  "F13.980": -1,
            "F13.981": -1,"F13.982": -1,"F13.988": -1, "F13.99": 0,
        },

        "F14": {  # Cocaine
            "F14.10": 0,  "F14.11": -1, "F14.120": -1, "F14.121": -1,
            "F14.122": -1,"F14.129": 0, "F14.13": -1,  "F14.14": -1,
            "F14.150": -1,"F14.151": -1,"F14.159": 0,  "F14.180": -1,
            "F14.181": -1,"F14.182": -1,"F14.188": -1, "F14.19": 0,
            "F14.20": 0,  "F14.21": -1, "F14.220": -1, "F14.221": -1,
            "F14.222": -1,"F14.229": 0, "F14.23": -1,  "F14.24": -1,
            "F14.250": -1,"F14.251": -1,"F14.259": 0,  "F14.280": -1,
            "F14.281": -1,"F14.282": -1,"F14.288": -1, "F14.29": 0,
            "F14.90": 0,  "F14.920": -1,"F14.921": -1, "F14.922": -1,
            "F14.929": 0, "F14.93": -1, "F14.94": -1,  "F14.950": -1,
            "F14.951": -1,"F14.959": 0, "F14.980": -1, "F14.981": -1,
            "F14.982": -1,"F14.988": -1,"F14.99": 0,
        },

        "F15": {  # Other stimulants (incl. caffeine, amphetamine)
            "F15.10": 0,  "F15.11": -1, "F15.120": -1, "F15.121": -1,
            "F15.122": -1,"F15.129": 0, "F15.13": -1,  "F15.14": -1,
            "F15.150": -1,"F15.151": -1,"F15.159": 0,  "F15.180": -1,
            "F15.181": -1,"F15.182": -1,"F15.188": -1, "F15.19": 0,
            "F15.20": 0,  "F15.21": -1, "F15.220": -1, "F15.221": -1,
            "F15.222": -1,"F15.229": 0, "F15.23": -1,  "F15.24": -1,
            "F15.250": -1,"F15.251": -1,"F15.259": 0,  "F15.280": -1,
            "F15.281": -1,"F15.282": -1,"F15.288": -1, "F15.29": 0,
            "F15.90": 0,  "F15.920": -1,"F15.921": -1, "F15.922": -1,
            "F15.929": 0, "F15.93": -1, "F15.94": -1,  "F15.950": -1,
            "F15.951": -1,"F15.959": 0, "F15.980": -1, "F15.981": -1,
            "F15.982": -1,"F15.988": -1,"F15.99": 0,
        },

        "F16": {  # Hallucinogens
            "F16.10": 0,  "F16.11": -1, "F16.120": -1, "F16.121": -1,
            "F16.122": -1,"F16.129": 0, "F16.13": -1,  "F16.14": -1,
            "F16.150": -1,"F16.151": -1,"F16.159": 0,  "F16.183": -1,
            "F16.188": -1,"F16.19": 0,
            "F16.20": 0,  "F16.21": -1, "F16.220": -1, "F16.221": -1,
            "F16.229": 0, "F16.24": -1, "F16.250": -1, "F16.251": -1,
            "F16.259": 0, "F16.283": -1,"F16.288": -1, "F16.29": 0,
            "F16.90": 0,  "F16.920": -1,"F16.921": -1, "F16.929": 0,
            "F16.94": -1, "F16.950": -1,"F16.951": -1, "F16.959": 0,
            "F16.983": -1,"F16.988": -1,"F16.99": 0,
        },

        "F17": {  # Nicotine
            "F17.200": 0, "F17.201": -1,"F17.203": -1, "F17.208": -1,
            "F17.209": 0,
            "F17.210": 0, "F17.211": -1,"F17.213": -1, "F17.218": -1,
            "F17.219": 0,
            "F17.220": 0, "F17.221": -1,"F17.223": -1, "F17.228": -1,
            "F17.229": 0,
            "F17.290": 0, "F17.291": -1,"F17.293": -1, "F17.298": -1,
            "F17.299": 0,
        },

        "F18": {  # Inhalants
            "F18.10": 0,  "F18.11": -1, "F18.120": -1, "F18.121": -1,
            "F18.129": 0, "F18.13": -1, "F18.14": -1,  "F18.17": -1,
            "F18.180": -1,"F18.188": -1,"F18.19": 0,
            "F18.20": 0,  "F18.21": -1, "F18.220": -1, "F18.221": -1,
            "F18.229": 0, "F18.23": -1, "F18.24": -1,  "F18.27": -1,
            "F18.280": -1,"F18.288": -1,"F18.29": 0,
            "F18.90": 0,  "F18.920": -1,"F18.921": -1, "F18.929": 0,
            "F18.94": -1, "F18.97": -1, "F18.980": -1, "F18.988": -1,
            "F18.99": 0,
        },

        "F19": {  # Other / multiple / unspecified psychoactive substances
            "F19.10": 0,  "F19.11": -1, "F19.120": -1, "F19.121": -1,
            "F19.122": -1,"F19.129": 0, "F19.13": -1,  "F19.14": -1,
            "F19.150": -1,"F19.151": -1,"F19.159": 0,  "F19.16": -1,
            "F19.17": -1, "F19.180": -1,"F19.181": -1, "F19.182": -1,
            "F19.188": -1,"F19.19": 0,
            "F19.20": 0,  "F19.21": -1, "F19.220": -1, "F19.221": -1,
            "F19.222": -1,"F19.229": 0, "F19.23": -1,  "F19.24": -1,
            "F19.250": -1,"F19.251": -1,"F19.259": 0,  "F19.26": -1,
            "F19.27": -1, "F19.280": -1,"F19.281": -1, "F19.282": -1,
            "F19.288": -1,"F19.29": 0,
            "F19.90": 0,  "F19.920": -1,"F19.921": -1, "F19.922": -1,
            "F19.929": 0, "F19.93": -1, "F19.94": -1,  "F19.950": -1,
            "F19.951": -1,"F19.959": 0, "F19.96": -1,  "F19.97": -1,
            "F19.980": -1,"F19.981": -1,"F19.982": -1, "F19.988": -1,
            "F19.99": 0,
        },
    },

    # ------------------------------------------------------------------ #
    # F20-F29  Schizophrenia, schizotypal, delusional, and other          #
    #          non-mood psychotic disorders                               #
    # ------------------------------------------------------------------ #
    "F20-F29": {

        "F20": {  # Schizophrenia - episode specifiers, not severity scale
            "F20.0":  -1,  # paranoid
            "F20.1":  -1,  # disorganized (hebephrenic)
            "F20.2":  -1,  # catatonic
            "F20.3":  -1,  # undifferentiated
            "F20.5":  -1,  # residual
            "F20.81": -1,  # schizophreniform disorder
            "F20.89": -1,  # other
            "F20.9":   0,  # unspecified
        },

        "F21":  -1,  # Schizotypal disorder

        "F22":  -1,  # Delusional disorders

        "F23":  -1,  # Brief psychotic disorder

        "F24":  -1,  # Shared psychotic disorder (folie à deux)

        "F25": {  # Schizoaffective disorders - type, not severity
            "F25.0":  -1,  # bipolar type
            "F25.1":  -1,  # depressive type
            "F25.8":  -1,  # other
            "F25.9":   0,  # unspecified
        },

        "F28":  -1,  # Other specified psychotic disorder
        "F29":   0,  # Unspecified psychosis NOS
    },

    # ------------------------------------------------------------------ #
    # F30-F39  Mood [affective] disorders                                 #
    # ------------------------------------------------------------------ #
    "F30-F39": {

        "F30": {  # Manic episode - severity encoded in digit
            "F30.10":  0,  # manic, unspecified
            "F30.11":  1,  # manic, mild
            "F30.12":  2,  # manic, moderate
            "F30.13":  3,  # manic, severe without psychotic features
            "F30.2":   4,  # manic, severe with psychotic features
            "F30.3":  -1,  # manic in partial remission (remission, not severity)
            "F30.4":  -1,  # manic in full remission
            "F30.8":  -1,  # other manic episodes
            "F30.9":   0,  # manic episode, unspecified
        },

        "F31": {  # Bipolar disorder - severity per current episode
            # Manic episodes
            "F31.0":  -1,  # BP current episode hypomanic
            "F31.10":  0,  # BP current episode manic, unspecified
            "F31.11":  1,  # BP current episode manic, mild
            "F31.12":  2,  # BP current episode manic, moderate
            "F31.13":  3,  # BP current episode manic, severe without psychotic
            "F31.2":   4,  # BP current episode manic, severe with psychotic
            # Depressive episodes
            "F31.30":  0,  # BP current episode depressed, unspecified
            "F31.31":  1,  # BP current episode depressed, mild
            "F31.32":  2,  # BP current episode depressed, moderate
            "F31.4":   3,  # BP current episode depressed, severe without psychotic
            "F31.5":   4,  # BP current episode depressed, severe with psychotic
            # Mixed / remission
            "F31.60":  0,  # BP current episode mixed, unspecified
            "F31.61":  1,  # BP current episode mixed, mild
            "F31.62":  2,  # BP current episode mixed, moderate
            "F31.63":  3,  # BP current episode mixed, severe without psychotic
            "F31.64":  4,  # BP current episode mixed, severe with psychotic
            "F31.70": -1,  # BP in partial remission, most recent episode unspecified
            "F31.71": -1,  # BP in partial remission, most recent episode hypomanic
            "F31.72": -1,  # BP in partial remission, most recent episode manic
            "F31.73": -1,  # BP in partial remission, most recent episode depressed
            "F31.74": -1,  # BP in partial remission, most recent episode mixed
            "F31.75": -1,  # BP in full remission, most recent episode unspecified
            "F31.76": -1,  # BP in full remission, most recent episode hypomanic
            "F31.77": -1,  # BP in full remission, most recent episode manic
            "F31.78": -1,  # BP in full remission, most recent episode depressed
            "F31.81": -1,  # Bipolar II disorder
            "F31.89": -1,  # Other bipolar disorder
            "F31.9":   0,  # Bipolar disorder, unspecified
        },

        "F32": {  # Major depressive disorder, single episode - severity encoded
            "F32.0":   1,  # mild
            "F32.1":   2,  # moderate
            "F32.2":   3,  # severe without psychotic features
            "F32.3":   4,  # severe with psychotic features
            "F32.4":  -1,  # in partial remission
            "F32.5":  -1,  # in full remission
            "F32.81": -1,  # premenstrual dysphoric disorder
            "F32.89": -1,  # other specified depressive episodes
            "F32.9":   0,  # unspecified
            "F32.A":  -1,  # depression, unspecified (alternate code)
        },

        "F33": {  # Major depressive disorder, recurrent - severity encoded
            "F33.0":   1,  # mild
            "F33.1":   2,  # moderate
            "F33.2":   3,  # severe without psychotic features
            "F33.3":   4,  # severe with psychotic features
            "F33.40": -1,  # in remission, unspecified
            "F33.41": -1,  # in partial remission
            "F33.42": -1,  # in full remission
            "F33.8":  -1,  # other
            "F33.9":   0,  # unspecified
        },

        "F34": {  # Persistent mood disorders - type, not severity
            "F34.0":  -1,  # cyclothymia
            "F34.1":  -1,  # dysthymia
            "F34.81": -1,  # disruptive mood dysregulation disorder
            "F34.89": -1,  # other persistent mood disorders
            "F34.9":   0,  # unspecified
        },

        "F39":  0,  # Unspecified mood disorder
    },

    # ------------------------------------------------------------------ #
    # F40-F48  Anxiety, dissociative, stress-related, somatoform and      #
    #          other nonpsychotic mental disorders                        #
    # ------------------------------------------------------------------ #
    "F40-F48": {

        "F40": {  # Phobic anxiety disorders
            "F40.00": 0,
            "F40.01": -1,
            "F40.02": -1,
            "F40.10": 0,
            "F40.11": -1,
            "F40.210": -1,
            "F40.218": -1,
            "F40.220": -1,
            "F40.228": -1,
            "F40.230": -1,
            "F40.231": -1,
            "F40.232": -1,
            "F40.233": -1,
            "F40.240": -1,
            "F40.241": -1,
            "F40.242": -1,
            "F40.243": -1,
            "F40.248": -1,
            "F40.290": -1,
            "F40.291": -1,
            "F40.298": -1,
            "F40.8":  -1,
            "F40.9":   0,
        },

        "F41": {  # Other anxiety disorders
            "F41.0":  -1,  # panic disorder
            "F41.1":  -1,  # GAD
            "F41.3":  -1,  # other mixed anxiety
            "F41.8":  -1,  # other specified
            "F41.9":   0,  # unspecified
        },

        "F42": {  # OCD
            "F42.2":  -1,
            "F42.3":  -1,
            "F42.4":  -1,
            "F42.8":  -1,
            "F42.9":   0,
        },

        "F43": {  # Reaction to severe stress and adjustment disorders
            "F43.0":  -1,  # acute stress reaction
            "F43.10":  0,  # PTSD unspecified
            "F43.11": -1,  # PTSD acute
            "F43.12": -1,  # PTSD chronic
            "F43.20":  0,  # adjustment disorder unspecified
            "F43.21": -1,  # with depressed mood
            "F43.22": -1,  # with anxiety
            "F43.23": -1,  # with mixed anxiety and depression
            "F43.24": -1,  # with conduct disturbance
            "F43.25": -1,  # with mixed disturbance of emotions and conduct
            "F43.29": -1,  # other
            "F43.8":  -1,
            "F43.9":   0,
        },

        "F44": {  # Dissociative and conversion disorders
            "F44.0":  -1,
            "F44.1":  -1,
            "F44.2":  -1,
            "F44.4":  -1,
            "F44.5":  -1,
            "F44.6":  -1,
            "F44.7":  -1,
            "F44.81": -1,
            "F44.89": -1,
            "F44.9":   0,
        },

        "F45": {  # Somatoform disorders
            "F45.0":  -1,
            "F45.1":  -1,
            "F45.20": 0,
            "F45.21": -1,
            "F45.22": -1,
            "F45.29": -1,
            "F45.41": -1,
            "F45.42": -1,
            "F45.8":  -1,
            "F45.9":   0,
        },

        "F48": {  # Other nonpsychotic mental disorders
            "F48.1":  -1,  # depersonalization-derealization syndrome
            "F48.2":  -1,  # pseudobulbar affect
            "F48.8":  -1,
            "F48.9":   0,
        },
    },

    # ------------------------------------------------------------------ #
    # F50-F59  Behavioural syndromes associated with physiological        #
    #          disturbances and physical factors                          #
    # ------------------------------------------------------------------ #
    "F50-F59": {

        "F50": {  # Eating disorders
            "F50.00":  0,
            "F50.01": -1,
            "F50.02": -1,
            "F50.09": -1,
            "F50.2":  -1,
            "F50.81": -1,
            "F50.82": -1,
            "F50.89": -1,
            "F50.9":   0,
        },

        "F51": {  # Non-organic sleep disorders
            "F51.01": -1,
            "F51.02": -1,
            "F51.03": -1,
            "F51.04": -1,
            "F51.05": -1,
            "F51.09": 0,
            "F51.11": -1,
            "F51.12": -1,
            "F51.13": -1,
            "F51.19": 0,
            "F51.3":  -1,
            "F51.4":  -1,
            "F51.5":  -1,
            "F51.8":  -1,
            "F51.9":   0,
        },

        "F52": {  # Sexual dysfunction not due to substance or known physiological condition
            "F52.0":  -1,
            "F52.1":  -1,
            "F52.21": -1,
            "F52.22": -1,
            "F52.31": -1,
            "F52.32": -1,
            "F52.4":  -1,
            "F52.5":  -1,
            "F52.6":  -1,
            "F52.8":  -1,
            "F52.9":   0,
        },

        "F53": {  # Mental and behavioural disorders associated with the puerperium
            "F53.0":  -1,
            "F53.1":  -1,
        },

        "F54":  -1,  # Psychological factors affecting physical conditions
        "F55": {
            "F55.0":  -1,
            "F55.1":  -1,
            "F55.2":  -1,
            "F55.3":  -1,
            "F55.4":  -1,
            "F55.5":  -1,
            "F55.6":  -1,
            "F55.8":  -1,
        },
        "F59":   0,
    },

    # ------------------------------------------------------------------ #
    # F60-F69  Disorders of adult personality and behaviour               #
    # ------------------------------------------------------------------ #
    "F60-F69": {

        "F60": {  # Specific personality disorders - type, not severity
            "F60.0":  -1,  # paranoid
            "F60.1":  -1,  # schizoid
            "F60.2":  -1,  # antisocial
            "F60.3":  -1,  # borderline
            "F60.4":  -1,  # histrionic
            "F60.5":  -1,  # obsessive-compulsive
            "F60.6":  -1,  # avoidant
            "F60.7":  -1,  # dependent
            "F60.81": -1,  # narcissistic
            "F60.89": -1,  # other
            "F60.9":   0,  # unspecified
        },

        "F63": {  # Impulse disorders
            "F63.0":  -1,
            "F63.1":  -1,
            "F63.2":  -1,
            "F63.3":  -1,
            "F63.81": -1,
            "F63.89": -1,
            "F63.9":   0,
        },

        "F64": {  # Gender identity disorders
            "F64.0":  -1,
            "F64.1":  -1,
            "F64.2":  -1,
            "F64.8":  -1,
            "F64.9":   0,
        },

        "F65": {  # Paraphilias
            "F65.0":  -1,
            "F65.1":  -1,
            "F65.2":  -1,
            "F65.3":  -1,
            "F65.4":  -1,
            "F65.50": 0,
            "F65.51": -1,
            "F65.52": -1,
            "F65.59": -1,
            "F65.81": -1,
            "F65.89": -1,
            "F65.9":   0,
        },

        "F66":  -1,  # Other sexual disorders

        "F68": {
            "F68.10": 0,
            "F68.11": -1,
            "F68.12": -1,
            "F68.13": -1,
            "F68.8":  -1,
            "F68.A":  -1,
        },

        "F69":   0,
    },

    # ------------------------------------------------------------------ #
    # F70-F79  Intellectual disabilities                                  #
    # ------------------------------------------------------------------ #
    "F70-F79": {
        # Severity IS explicitly encoded by the leading digit here
        "F70":  1,  # mild intellectual disability
        "F71":  2,  # moderate intellectual disability
        "F72":  3,  # severe intellectual disability
        "F73":  4,  # profound intellectual disability
        "F78": {
            "F78.A1": -1,
            "F78.A9": -1,
        },
        "F79":  0,  # unspecified intellectual disability
    },

    # ------------------------------------------------------------------ #
    # F80-F89  Pervasive and specific developmental disorders             #
    # ------------------------------------------------------------------ #
    "F80-F89": {

        "F80": {  # Developmental disorders of speech and language
            "F80.0":  -1,
            "F80.1":  -1,
            "F80.2":  -1,
            "F80.4":  -1,
            "F80.81": -1,
            "F80.82": -1,
            "F80.89": -1,
            "F80.9":   0,
        },

        "F81": {  # Developmental disorders of scholastic skills
            "F81.0":  -1,
            "F81.2":  -1,
            "F81.81": -1,
            "F81.89": -1,
            "F81.9":   0,
        },

        "F82":  -1,  # Developmental disorder of motor function (DCD)

        "F84": {  # Pervasive developmental disorders
            "F84.0":  -1,  # autism spectrum disorder
            "F84.2":  -1,  # Rett syndrome
            "F84.3":  -1,  # childhood disintegrative disorder
            "F84.5":  -1,  # Asperger syndrome
            "F84.8":  -1,
            "F84.9":   0,
        },

        "F88":  -1,  # Other disorders of psychological development
        "F89":   0,  # Unspecified disorder of psychological development
    },

    # ------------------------------------------------------------------ #
    # F90-F98  Behavioural and emotional disorders with onset usually     #
    #          occurring in childhood and adolescence                     #
    # ------------------------------------------------------------------ #
    "F90-F98": {

        "F90": {  # ADHD - presentation type, not severity
            "F90.0":  -1,  # inattentive type
            "F90.1":  -1,  # hyperactive type
            "F90.2":  -1,  # combined type
            "F90.8":  -1,  # other
            "F90.9":   0,  # unspecified
        },

        "F91": {  # Conduct disorders
            "F91.0":  -1,
            "F91.1":  -1,
            "F91.2":  -1,
            "F91.3":  -1,
            "F91.8":  -1,
            "F91.9":   0,
        },

        "F93": {  # Emotional disorders with onset specific to childhood
            "F93.0":  -1,
            "F93.8":  -1,
            "F93.9":   0,
        },

        "F94": {  # Disorders of social functioning with onset specific to childhood
            "F94.0":  -1,
            "F94.1":  -1,
            "F94.2":  -1,
            "F94.8":  -1,
            "F94.9":   0,
        },

        "F95": {  # Tic disorders
            "F95.0":  -1,
            "F95.1":  -1,
            "F95.2":  -1,
            "F95.8":  -1,
            "F95.9":   0,
        },

        "F98": {  # Other behavioural and emotional disorders
            "F98.0":  -1,
            "F98.1":  -1,
            "F98.21": -1,
            "F98.29": -1,
            "F98.3":  -1,
            "F98.4":  -1,
            "F98.5":  -1,
            "F98.8":  -1,
            "F98.9":   0,
        },
    },

    # ------------------------------------------------------------------ #
    # F99  Unspecified mental disorder                                    #
    # ------------------------------------------------------------------ #
    "F99-F99": {
        "F99": 0,
    },
}

def _build_severity_lookup() -> dict[str, int]:
    """Flatten nested ICD10_F_SEVERITY"""
    lookup: dict[str, int] = {}
    def _walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            else:
                lookup[k] = v
    _walk(ICD10_F_SEVERITY)
    return lookup

SEVERITY_LOOKUP: dict[str, int] = _build_severity_lookup()

# =============================================================================
# Set of codes indicating remission, which should be ignored when determining severity
REMISSION_CODES: frozenset[str] = frozenset({
    "F30.3", "F30.4",
    "F31.70", "F31.71", "F31.72", "F31.73", "F31.74",
    "F31.75", "F31.76", "F31.77", "F31.78",
    "F32.4",  "F32.5",
    "F33.40", "F33.41", "F33.42",
})



