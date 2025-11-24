'''
File: settings.py
Author: Adam Pah
Description: Settings file
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORE_DATA = PROJECT_ROOT / 'code'/ 'support' / 'core_data'
DATAPATH = PROJECT_ROOT / 'data'
ANNO_PATH = DATAPATH / 'annotation'
PACER_PATH = DATAPATH / 'pacer' # generate using scrapers.py

COURTFILE = CORE_DATA / 'district_courts.csv'
DISTRICT_COURTS_94 = CORE_DATA / 'district_courts_94.csv'
STATEY2CODE = CORE_DATA / 'statey2code.json'
NATURE_SUIT = CORE_DATA / 'nature_suit.csv'
JUDGEFILE = CORE_DATA / 'judge_demographics.csv'
BAMAG_JUDGES = CORE_DATA / 'brmag_judges.csv'
BAMAG_POSITIONS = CORE_DATA / 'brmag_positions.csv'

MEM_DF = DATAPATH / 'member_cases.csv'
LOG_DIR = DATAPATH / 'logs'
EXCLUDE_CASES = DATAPATH / 'exclude.csv'
UNIQUE_FILES_TABLE = DATAPATH / 'unique_docket_filepaths_table.csv' # generate using generate_unique_filepaths in data_tools.py
FJC =  DATAPATH / 'fjc' # generate using fjc.gov/research/idb and fjc_functions.py

MEMBER_LEAD_LINKS = ANNO_PATH / 'member_lead_links.jsonl'
ROLE_MAPPINGS = ANNO_PATH / 'role_mappings.json'
JEL_JSONL = ANNO_PATH / 'judge_disambiguation' / 'JEL.jsonl' # generate using the Research-Materials repo
ONTOLOGY_LABELS = ANNO_PATH / 'ontology' / 'labels.csv' # generate using the scales-nlp repo

ANNO_PATH_CLAYTON = ANNO_PATH / 'counties' / 'ga_clayton'
NIBRS_CATEGORIES_CLAYTON = ANNO_PATH_CLAYTON / 'nibrs' / 'nibrs_categories.csv'
NIBRS_CROSSWALK_CLAYTON = ANNO_PATH_CLAYTON / 'nibrs' / 'nibrs_crosswalk.csv'

# included on behalf of make_graph_data_pacer.py
# (in infrastructure_dev, this is a dev/prod switch, but it's not included here because pacer-tools is always prod)
use_datastore = lambda path: path
