import re
import ast
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rdflib import Graph, URIRef

from constants import SCALES

manual_offense_mapping = {
    "ASSAULT AGGRAVATED": "AGGRAVATED ASSAULT",
    "ASSAULT SIMPLE": "SIMPLE ASSAULT",
    "INTIMIDATION": "INTIMIDATION",
    "DRUG PARAPHERNALIA OFFENSES": "DRUG EQUIPMENT VIOLATIONS",
    "EQUIPMENT DRUG": "DRUG EQUIPMENT VIOLATIONS",
    "FALSE PRETENSES": "FALSE PRETENSES/SWINDLE/CONFIDENCE GAME",
    "SWINDLE": "FALSE PRETENSES/SWINDLE/CONFIDENCE GAME",
    "CONFIDENCE GAME": "FALSE PRETENSES/SWINDLE/CONFIDENCE GAME",
    "AUTOMATED TELLER MACHINE": "CREDIT CARD/AUTOMATED TELLER MACHINE FRAUD",
    "CREDIT CARD FRAUD": "CREDIT CARD/AUTOMATED TELLER MACHINE FRAUD",
    "IMPERSONATION": "IMPERSONATION",
    "FRAUD WELFARE": "WELFARE FRAUD",
    "FRAUD TELEPHONE": "WIRE FRAUD",
    "FRAUD IDENTITY THEFT": "IDENTITY THEFT",
    "COMPUTER CRIME": "HACKING/COMPUTER INVASION",
    "FRAUD HACKING/COMPUTER\nINVASION": "HACKING/COMPUTER INVASION",
    "BETTING UNLAWFUL": "BETTING/WAGERING",
    "TRANSMITTING WAGERING INFORMATION": "BETTING/WAGERING",
    "WAGERING UNLAWFUL": "BETTING/WAGERING",
    "GAMBLING PARAPHERNALIA DEVICES EQUIPMENT POSESSION": "GAMBLING EQUIPMENT VIOLATIONS",
    "BRIBERY SPORTS": "SPORTS TAMPERING",
    "HOMICIDE JUSTIFIABLE": "JUSTIFIABLE HOMICIDE",
    "COMMERCIALIZED SEX COMMERCIAL SEX": "HUMAN TRAFFICKING, COMMERCIAL SEX ACTS",
    "HUMAN TRAFFICKING\nCOMMERCIAL SEX ACTS": "HUMAN TRAFFICKING, COMMERCIAL SEX ACTS",
    "PICKPOCKET": "POCKET-PICKING",
    "PURSE-SNATCHING": "PURSE-SNATCHING",
    "SHOPLIFTING": "SHOPLIFTING",
    "THEFT FROM A BUILDING": "THEFT FROM BUILDING",
    "THEFT FROM A COIN-OPERATED\nMACHINE OR DEVICE": "THEFT FROM COIN-OPERATED MACHINE OR DEVICE",
    "THEFT FROM A MOTOR VEHICLE": "THEFT FROM MOTOR VEHICLE",
    "STRIPPING MOTOR VEHICLE": "THEFT OF MOTOR VEHICLE PARTS OR ACCESSORIES",
    "PIMPING": "ASSISTING OR PROMOTING PROSTITUTION",
    "TRANSPORTING PERSONS FOR PROSTITUTION": "ASSISTING OR PROMOTING PROSTITUTION",
    "FREQUENTING A HOUSE OF\nPROSTITUTION": "PURCHASING PROSTITUTION",
    "RAPE": "RAPE",
    "SODOMY": "SODOMY",
    "SEXUAL ASSAULT WITH AN OBJECT": "SEXUAL ASSAULT WITH AN OBJECT",
    "FONDLING": "FONDLING",
    "INCEST": "INCEST",
    "RAPE STATUTORY": "STATUTORY RAPE",
    "EXPLOSIVES": "EXPLOSIVES",
}

drug_keywords_apd = {
    "crack": {"nibrs_code": "A", "nibrs_drug": "Crack Cocaine"},
    "caine": {"nibrs_code": "B", "nibrs_drug": "Cocaine (All forms except crack)"},
    "hash": {"nibrs_code": "C", "nibrs_drug": "Hashish"},
    "roin": {"nibrs_code": "D", "nibrs_drug": "Heroin"},
    "juana": {"nibrs_code": "E", "nibrs_drug": "Marijuana"},
    "morp": {"nibrs_code": "F", "nibrs_drug": "Morphine"},
    # 'opium': {'nibrs_code': 'G', 'nibrs_drug': 'Opium'},
    "narc": {"nibrs_code": "H", "nibrs_drug": "Other Narcotics"},
    "lsd": {"nibrs_code": "I", "nibrs_drug": "LSD"},
    "pcp": {"nibrs_code": "J", "nibrs_drug": "PCP"},
    "halluc": {"nibrs_code": "K", "nibrs_drug": "Other Hallucinogens"},
    "amphe": {"nibrs_code": "L", "nibrs_drug": "Amphetamines/Methamphetamines"},
    "stim": {"nibrs_code": "M", "nibrs_drug": "Other Stimulants"},
    "barbit": {"nibrs_code": "N", "nibrs_drug": "Barbiturates"},
    "depress": {"nibrs_code": "O", "nibrs_drug": "Other Depressants"},
    "unknown": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "drug": {"nibrs_code": "P", "nibrs_drug": "Other Drugs"},
    # 'over 3': 'X'
}
exclusions_apd = ()

drug_keywords_clayton = {
    "cocaine": {"nibrs_code": "B", "nibrs_drug": "Cocaine (All forms except crack)"},
    "substance or marijuana": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "substance/marijuana": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "marijuana": {"nibrs_code": "E", "nibrs_drug": "Marijuana"},
    "thc": {"nibrs_code": "E", "nibrs_drug": "Marijuana"},
    "ecstacy": {"nibrs_code": "K", "nibrs_drug": "Other Hallucinogens"},
    "amphetamine": {"nibrs_code": "L", "nibrs_drug": "Amphetamines/Methamphetamines"},
    "methaqualone": {"nibrs_code": "O", "nibrs_drug": "Other Depressants"},
    "ephedrine": {"nibrs_code": "P", "nibrs_drug": "Other Drugs"},
    "glue": {"nibrs_code": "P", "nibrs_drug": "Other Drugs"},
    "nitrous": {"nibrs_code": "P", "nibrs_drug": "Other Drugs"},
    "steroid": {"nibrs_code": "P", "nibrs_drug": "Other Drugs"},
    "drug": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "narcotic": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "gcsa": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "substa": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "medication": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
    "morphine, opium, heroin": {"nibrs_code": "U", "nibrs_drug": "Unknown Drug Type"},
}
exclusions_clayton = (
    "alcohol-drugs",
    "drug related object",
    "drugs/alcohol or under influence",
    "drugs,alcohol",
    "drugs, weapons or alcohol",
    "dumping",
)



def process_entities(fpaths, outdir, party_uri_func, fields_needed, filter_funcs={}):
    '''
    fpaths: an iterable of filepaths from which to extract entity info
    party_uri_func: a function with which to generate a party uri for each dataframe row (not an entity uri, as df.id will be used for these by default)
    fields_needed: an iterable of fields that party_uri_func needs (i believe this is more performant than using iterrows)
    filters: optionally, a dict that maps each desired filepath to a lambda function that filters a dataframe (e.g. to exclude weak keys or select courts in PARTY_DIS_UNIVERSAL)
    '''
    g = Graph()
    g.bind('scales', SCALES)

    for fpath in fpaths:
        df = pd.read_csv(fpath)
        filter_func = filter_funcs.get(fpath)
        if filter_func:
            df = filter_func(df)
        if 'id' not in df.columns:
            raise Exception(f"process_entities expects {fpath} to contain an 'id' column")

        spids = list(df.id)
        field_lists = [list(df[field]) for field in fields_needed]
        for i in tqdm(range(len(df)), desc='Processing disambiguated parties'):
            values = [lst[i] for lst in field_lists]
            g.add((party_uri_func(*values), SCALES.isInstanceOfEntity, _make_generic_uri('PartyEntity', spids[i])))

            if i and not i%50000:
                _write_graph_to_file(g, outdir, infix="entities")
                g = Graph()
                g.bind('scales', SCALES)

    # TODO merge ids in a more conservative disambiguation file when a more liberal disambiguation file suggests we can
    _write_graph_to_file(g, outdir, infix="entities")


def parse_drugs(df, charge_col, source, from_cli=False):
    results = []
    processed_indices = set()
    drug_keywords = {"apd": drug_keywords_apd, "clayton": drug_keywords_clayton}[source]
    exclusions = {"apd": exclusions_apd, "clayton": exclusions_clayton}[source]

    for index, row in df.iterrows():
        if index in processed_indices:
            continue
        arrest_charge = str(row[charge_col])
        for keyword, code in drug_keywords.items():
            if keyword.lower() in arrest_charge.lower() and not any(
                x in arrest_charge.lower() for x in exclusions
            ):
                results.append(
                    {
                        "index": index,
                        charge_col: arrest_charge,
                        "keyword": keyword,
                        "nibrs_code": code["nibrs_code"],
                        "nibrs_drug": code["nibrs_drug"],
                    }
                )
                processed_indices.add(index)
                break  # exit inner loop once a match is found for this record

    results_df = pd.DataFrame(results)
    if from_cli:
        results_df.to_csv("apd_drug_arrests.csv", index=False)
        print(f"Total arrest records processed: {len(df):,}")
        print(f"Total matches found: {len(results_df):,}")
        if not results_df.empty:
            print("\nNIBRS drugs by match count:\n")
            top_drugs = results_df["nibrs_drug"].value_counts().head(10)
            for drug, count in top_drugs.items():
                # Find the keyword for this drug
                keyword = results_df[results_df["nibrs_drug"] == drug]["keyword"].iloc[
                    0
                ]
                print(f"  {drug}, {keyword}: {count}")

                drug_matches = results_df[results_df["nibrs_drug"] == drug]
                top_charges = drug_matches[charge_col].value_counts()
                for charge, charge_count in top_charges.items():
                    print(f"    - {charge}: {charge_count}")
                print()
        else:
            print("No matches found.")
    else:
        return results_df


def _escape_quotes(text):
    if text is None:
        return None

    text = str(text)
    if '"' in text:
        return text.replace('"', "'")
    return text


def _date_to_xsd(date_str):
    if not date_str:
        return None

    # convert to string and strip whitespace
    date_str = str(date_str).strip()

    # 1) Fast-path: leading ISO YYYY-MM-DD (optionally followed by time info)
    if len(date_str) >= 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str[:10]

    # 2) Try a list of known patterns via time.strptime
    patterns = [
        "%Y-%m",  # '2016-03'
        "%m/%d/%Y",  # '03/12/2016'
        "%d/%m/%Y",  # '12/03/2016' (rare)
        "%m/%Y",  # '03/2016'
    ]

    for fmt in patterns:
        try:
            parsed = time.strptime(date_str, fmt)
            # Default missing day/month handled by strptime (defaults to 1)
            return time.strftime("%Y-%m-%d", parsed)
        except ValueError:
            pass
            # raise ValueError(f"Invalid date format: {date_str}")


def _make_case_uri(ucid):
    return URIRef(f"{SCALES}Case/{ucid}")

def _make_docket_uri(ucid, idx):
    return URIRef(f"{SCALES}DocketEntry/{ucid}_de{int(idx)}")

def _make_charge_uri(ucid, dft_idx, chg_idx):
    if type(chg_idx)==str:
        chg_idx = re.sub('[ :;,./="]', "", chg_idx)
    return URIRef(f"{SCALES}Charge/{ucid}_p{int(dft_idx)}_c{chg_idx}")

def _make_sentence_uri(ucid, entry_idx, sentence_idx):
    return URIRef(f"{SCALES}Sentence/{ucid}_de{int(entry_idx)}_s{int(sentence_idx)}")

def _make_party_uri(ucid, idx):
    return URIRef(f"{SCALES}Party/{ucid}_p{int(idx)}")

def _make_counsel_uri(ucid, idx):
    return URIRef(f"{SCALES}Lawyer/{ucid}_l{int(idx)}")

def _make_generic_uri(namespace, entity_id):
    return URIRef(f"{SCALES}{namespace}/{entity_id}")


def _write_graph_to_file(graph, outdir, file_name=None, infix=None):
    """Write the current graph to a file with a unique, sortable name."""
    file_name = file_name or f"graph_{infix+'_' if infix else ''}{time.time_ns()}.ttl"
    outpath = Path(outdir) / Path(file_name)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing TTL to {outpath}")
    graph.serialize(destination=str(outpath), format="turtle", encoding="utf-8")
    print(f"Wrote TTL to {outpath}")
