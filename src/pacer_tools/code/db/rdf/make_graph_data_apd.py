#!/usr/bin/env python3

"""
This script has been included solely in an effort to make our RDF-graph-build process
transparent; it was copy-pasted directly from SCALES's private infrastructure repo, and
has not been tested here! Our assumption is that, because the raw data used for this
portion of our graph comes from a private dataset, nobody besides us will run this
script. If we're incorrect about this assumption, feel free to contact us at
engineering@scales-okn.org.
"""

"""Convert Atlanta Police Department CSV arrest data to RDF Turtle files.

This implementation mirrors the Clayton-county and refactored Fulton-county
builders: unified namespace bindings, multiprocessing for speed, chunked TTL
output, and consistent NIEM/JXDM predicates.
"""

# --------------------------------------------------------------------------------------
# Standard library imports
# --------------------------------------------------------------------------------------

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import rdflib
from rdflib import Graph, Literal, Namespace, URIRef, RDF, XSD
from tqdm import tqdm
import zipfile
import geopandas
from pyproj import CRS
from shapely.geometry import Point
from math import radians, sin, cos, sqrt, atan2

# --------------------------------------------------------------------------------------
# Local dependencies
# --------------------------------------------------------------------------------------

sys.path.append(str(Path.cwd().parents[1].resolve()))
import utils  # noqa: E402  pylint: disable=wrong-import-position
from constants import SCALES, J, NC, FIPS, NIBRS, TREATMENT
from support import settings

# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_handler = logging.FileHandler("error.log")
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(_handler)


# NIBRS
nibrs_crosswalk_dict = pd.read_csv(settings.NIBRS_CROSSWALK_APD).rename(
    lambda x: x.lower(), axis=1)
_format_nibrs_code = lambda x: {
    "A": "GROUP A INCIDENT REPORT",
    "B": "GROUP B ARREST REPORT",
}[x]
_format_officer_id = (
    lambda x: x.replace(" ", "_").replace("/", "_").replace(",", "-").replace("`", "")
)

# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------


def _haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8 # radius of Earth in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def _calculate_block_group_distance(block_group_id1, block_group_id2, gdf, is_old_gdf=False):
    if is_old_gdf:
        gdf.columns = [x.replace('10','') for x in gdf.columns]
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    bg1, bg2 = gdf[gdf['GEOID'] == str(block_group_id1)], gdf[gdf['GEOID'] == str(block_group_id2)]
    return _haversine_distance(float(bg1.INTPTLAT.iloc[0]), float(bg1.INTPTLON.iloc[0]),
                              float(bg2.INTPTLAT.iloc[0]), float(bg2.INTPTLON.iloc[0]))

def _create_graph() -> Graph:
    """Return a fresh graph with bound prefixes."""
    g = Graph()
    g.bind("scales", SCALES)
    g.bind("j", J)
    g.bind("nc", NC)
    g.bind("fips", FIPS)
    g.bind("nibrs", NIBRS)
    g.bind("treatment", TREATMENT)
    g.bind("rdf", RDF)
    return g


def _make_metadata_graph():
    """Take care of a few APD-related triples we might want to use at some point."""
    g = _create_graph()
    agency_uri = utils._make_generic_uri('PoliceDepartment', 'ga-atlanta-pd')
    g.add((agency_uri, NC.OrganizationName, 'Atlanta Police Department'))
    g.add((agency_uri, J.OrganizationCategoryNLETSCode, 'PD')) # see https://niem.github.io/model/5.0/nlets/OrganizationCategoryCodeSimpleType/#diagram
    g.add((agency_uri, NC.LocationCityName, 'Atlanta'))
    g.add((agency_uri, NC.LocationStateName, 'Georgia'))
    return g


# --------------------------------------------------------------------------------------
# URI helpers (based on utils._make_generic_uri)
# --------------------------------------------------------------------------------------


def _make_incident_uri(incident_id: str):
    return utils._make_generic_uri("Incident", f"ga-atlanta-pd-{incident_id}")


def _make_arrest_uri(offense_id: str):
    return utils._make_generic_uri("Arrest", f"ga-atlanta-pd-{offense_id}")


def _make_person_uri(offense_id: str):
    return utils._make_generic_uri("ArrestSubject", f"ga-atlanta-pd-{offense_id}")


def _make_charge_uri(offense_id: str):
    return utils._make_generic_uri("ArrestCharge", f"ga-atlanta-pd-{offense_id}")


def _make_officer_uri(officer_id: str):
    return utils._make_generic_uri(
        "PoliceOfficer", f"ga-atlanta-pd-{_format_officer_id(officer_id)}"
    )


# --------------------------------------------------------------------------------------
# Supplemental triples (added by Scott; originally designed for NSF Y3 presentation)
# --------------------------------------------------------------------------------------


def _add_supplemental_triples(g, apd_df):
    fdir = '/mnt/datastore/non_courts/apd/files_for_graph_build'

    # from FRINK
    block_groups_scales = set(pd.read_csv(f'{fdir}/block_groups_scales.csv')['block_group'])

    # from FRINK + geocoder call from remote-notebook archive 0
    _row_to_block_group = lambda x: int(str(x[8])+(
        '0'*(3-len(str(x[9]))))+str(x[9])+(
        '0'*(6-len(str(x[10]))))+str(x[10])+str(x[11])[:1])
    geocoder_df = pd.read_csv(f'{fdir}/block_groups_ruralkg.csv', header=None)
    block_groups_ruralkg = set([_row_to_block_group(x) for _,x in geocoder_df.iterrows()])

    # from FRINK (could've done this in the above block, but i didn't think of it at first)
    uris_addresses = pd.read_csv(f'{fdir}/sparql_results_ruralkg_with_id.csv')
    group_to_ruralkg_uri = {_row_to_block_group(geocoder_df[geocoder_df[1].str.contains(
        x['address'])].iloc[0]):x['provider'].strip('<>') for _,x in uris_addresses.iterrows()}


    nghb_df = pd.read_csv(f'{fdir}/neighborhoods.csv')
    nghb_df.columns = ['neighborhood', 'group']
    group_to_nghb = {x:list(nghb_df[nghb_df.group.eq(x)].neighborhood) for x in set(nghb_df.group)}


    groups_seen, hashes_seen, dists_seen = set(), set(), set()
    blockgroup_to_composite_uri = {} # part of the efficiency hack in the next cell
    gdf = geopandas.read_file(f'{fdir}/shapefiles/tl_2020_13_bg.shp')

    print('adding supplementary triples (main loop)...')
    for group1_orig in tqdm(block_groups_ruralkg):
        ruralkg_uri = URIRef(group_to_ruralkg_uri[group1_orig])
        g.add((ruralkg_uri, NC.LocaleCensusBlockID, Literal(str(group1_orig))))
        for group2 in block_groups_scales:
            group1 = group1_orig
            
            if group1>group2:
                temp = group1
                group1 = group2
                group2 = temp
            hsh = f'{group1}_{group2}'
            if hsh in hashes_seen:
                continue
            hashes_seen.add(hsh)
            
            dist = _calculate_block_group_distance(str(group1), str(group2), gdf)
            dist_uri = utils._make_generic_uri('DistanceMiles', dist)
            if dist not in dists_seen:
                dists_seen.add(dist)
                g.add((dist_uri, RDF.type, NC.RelativeLocationDistanceMeasure))
                g.add((dist_uri, NC.MeasureDecimalValue, Literal(dist, datatype=XSD.decimal)))
                g.add((dist_uri, NC.MeasureUnitText, Literal("mile")))
            
            for i,group in enumerate((group1, group2)):
                group_other = group1 if i else group2
                location_uri = utils._make_generic_uri('Location', group)
                location_relative_uri = utils._make_generic_uri('RelativeLocation', f'{group}_{group_other}')
                if group_other not in blockgroup_to_composite_uri:
                    blockgroup_to_composite_uri[group_other] = set()
                blockgroup_to_composite_uri[group_other].add(location_relative_uri)
                
                if group not in groups_seen:
                    groups_seen.add(group)
                    g.add((location_uri, RDF.type, NC.RelativeLocationReferencePoint))
                    g.add((location_uri, NC.LocaleCensusBlockID, Literal(str(group))))
                    for nghb in group_to_nghb.get(group, []):
                        g.add((location_uri, SCALES.containsNeighborhood, Literal(nghb)))
                g.add((location_relative_uri, RDF.type, NC.RelativeLocation))
                g.add((location_uri, NC.RelativeLocation, location_relative_uri))
                g.add((location_relative_uri, NC.RelativeLocationReferencePoint, location_uri))
                g.add((location_relative_uri, NC.LocaleCensusBlockID, Literal(str(group))))
                g.add((location_relative_uri, NC.RelativeLocationDistanceMeasure, dist_uri))

    # sparql-efficiency hack
    print('adding Location->TreatmentProvider triples...')
    for group1_orig in tqdm(block_groups_ruralkg):
        g.add((utils._make_generic_uri('Location', group1_orig), TREATMENT.TreatmentProvider, URIRef(group_to_ruralkg_uri[group1_orig])))
    print('adding RelativeLocation->Arrest triples...')
    for i,row in tqdm(apd_df.iterrows(), total=len(apd_df)):
        group = row['census_block_group_2020']
        if not pd.isna(group):
            for composite_uri in blockgroup_to_composite_uri[int(group)]:
                g.add((composite_uri, J.Arrest, utils._make_generic_uri('Arrest', f'ga-atlanta-pd-{i}')))
    return g


# --------------------------------------------------------------------------------------
# Row → triples conversion
# --------------------------------------------------------------------------------------


def _row_to_triples(
    row: Dict[str, Any], idx
) -> List[tuple]:  # noqa: C901 – complexity acceptable
    """Convert one CSV row (as dict) to a list of RDF triples."""
    g = _create_graph()

    incident_id = row.get("offense_id")
    if not incident_id:
        return []

    arrest_uri = _make_arrest_uri(str(idx))
    person_uri = _make_person_uri(str(idx))
    charge_uri = _make_charge_uri(str(idx))

    # from scott: i added this block because offense_id isn't unique (rather, it corresponds to an incident that entails 1 or more arrests)
    incident_uri = _make_incident_uri(incident_id)
    g.add((incident_uri, RDF.type, NC.Incident))
    g.add((incident_uri, J.IncidentArrest, arrest_uri))

    # -------------------------------------------------------------------------
    # Arrest event
    # -------------------------------------------------------------------------
    g.add((arrest_uri, RDF.type, J.Arrest))
    g.add((arrest_uri, J.ArrestAgency, utils._make_generic_uri('PoliceDepartment', 'ga-atlanta-pd')))
    g.add((arrest_uri, J.ArrestSubject, person_uri))
    g.add((arrest_uri, J.ArrestCharge, charge_uri))

    poss_date = row.get("poss_date") or row.get("arrest_date")
    if poss_date and str(poss_date).lower() != "nan":
        g.add(
            (
                arrest_uri,
                NC.ActivityDate,
                Literal(utils._date_to_xsd(poss_date), datatype=XSD.date),
            )
        )

    if (
        row.get("census_block_group_2020")
        and str(row["census_block_group_2020"]).lower() != "nan"
    ):
        g.add(
            (
                arrest_uri,
                NC.LocaleCensusBlockID,
                Literal(row["census_block_group_2020"]),
            )
        )

    # Officer information
    if row.get("rpt_officer_id") and str(row["rpt_officer_id"]).lower() != "nan":
        officer_id = row["rpt_officer_id"]
        officer_uri = _make_officer_uri(officer_id)
        g.add((arrest_uri, J.ArrestOfficial, officer_uri))
        g.add((officer_uri, RDF.type, J.LawEnforcementOfficial))
        g.add((officer_uri, NC.PersonID, Literal(officer_id)))

    # -------------------------------------------------------------------------
    # Person (arrestee)
    # -------------------------------------------------------------------------
    g.add((person_uri, RDF.type, NC.Person))

    if row.get("birth_year") and str(row["birth_year"]).lower() != "nan":
        g.add(
            (
                person_uri,
                SCALES.BirthYear,
                Literal(str(row["birth_year"]), datatype=XSD.gYear),
            )
        )

    if row.get("race") and str(row["race"]).lower() != "nan":
        g.add((person_uri, NC.PersonRaceText, Literal(row["race"])))

    if row.get("sex") and str(row["sex"]).lower() != "nan":
        g.add((person_uri, J.PersonSexCode, Literal(row["sex"])))

    # -------------------------------------------------------------------------
    # Charge
    # -------------------------------------------------------------------------
    g.add((charge_uri, RDF.type, J.ArrestCharge))

    if row.get("Arrest Charge") and str(row["Arrest Charge"]).lower() != "nan":
        g.add(
            (
                charge_uri,
                J.ChargeText,
                Literal(utils._escape_quotes(row["Arrest Charge"])),
            )
        )

    if row.get("charge_type") and str(row["charge_type"]).lower() != "nan":
        g.add(
            (charge_uri, J.ChargeDispositionCategoryText, Literal(row["charge_type"]))
        )

    if row.get("charge_level") and str(row["charge_level"]).lower() != "nan":
        g.add((charge_uri, J.ChargeSeverityLevelCode, Literal(row["charge_level"])))

    # NIBRS
    if incident_id and str(incident_id).lower() != "nan":
        # Look up NIBRS information from crosswalk
        # Convert both to strings to ensure proper matching
        nibrs_match = nibrs_crosswalk_dict[
            nibrs_crosswalk_dict["offense_id"].astype(str) == str(incident_id)
        ]

        if not nibrs_match.empty:
            nibrs_row = nibrs_match.iloc[0]

            # Add NIBRS offense if available
            if (
                nibrs_row.get("nibrs_offense")
                and str(nibrs_row["nibrs_offense"]).lower() != "nan"
            ):
                g.add(
                    (
                        charge_uri,
                        NIBRS.OffenseUCRCode,
                        Literal(nibrs_row["nibrs_offense"]),
                    )
                )

            # Add NIBRS category if available
            if (
                nibrs_row.get("crime_category")
                and str(nibrs_row["crime_category"]).lower() != "nan"
            ):
                g.add(
                    (
                        charge_uri,
                        NIBRS.OffenseUCRCode,
                        Literal(nibrs_row["crime_category"]),
                    )
                )

            # Add NIBRS A/B classification
            if nibrs_row.get("aorb") and str(nibrs_row["aorb"]).lower() != "nan":
                for code in nibrs_row["aorb"].split(" OR "):
                    g.add(
                        (
                            charge_uri,
                            NIBRS.NIBRSReportCategoryCode,
                            Literal(_format_nibrs_code(code)),
                        )
                    )

    return list(g)


# --------------------------------------------------------------------------------------
# File-writing helper
# --------------------------------------------------------------------------------------


def _write_graph_worker(graph: Graph, outdir: Path, file_name: Union[str, None] = None):
    utils._write_graph_to_file(graph, outdir, file_name=file_name)


# --------------------------------------------------------------------------------------
# Main processing routine
# --------------------------------------------------------------------------------------


def main(csv_path: str, outdir: str, chunk_size: int = 10_000):
    """Convert APD CSV arrest data to Turtle graphs, chunking every *chunk_size* rows."""

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    total_rows = len(df)

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    utils._write_graph_to_file(_make_metadata_graph(), outdir, file_name='arrestagency.ttl')

    global_graph = _create_graph()
    record_counter = 0
    write_futures: List[Any] = []

    with ProcessPoolExecutor(max_workers=12) as proc_exec, ThreadPoolExecutor(
        max_workers=8
    ) as thread_exec:
        futures = {
            proc_exec.submit(_row_to_triples, row.to_dict(), idx): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=total_rows, desc="Processing arrests") as pbar:
            for future in as_completed(futures):
                triples = future.result()
                if triples:
                    for triple in triples:
                        global_graph.add(triple)
                    record_counter += 1

                    if record_counter >= chunk_size:
                        wf = thread_exec.submit(
                            _write_graph_worker, global_graph, outdir_p
                        )
                        write_futures.append(wf)
                        global_graph = _create_graph()
                        record_counter = 0
                pbar.update(1)

    global_graph = _add_supplemental_triples(global_graph, df)

    # Flush remainder
    if record_counter:
        utils._write_graph_to_file(global_graph, outdir_p)

    # Ensure all writes complete
    for wf in as_completed(write_futures):
        try:
            wf.result()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Error in write operation: %s", exc)


# --------------------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Atlanta Police Department CSV arrest data to Turtle graphs using the SCALES RDF schema.",
    )
    parser.add_argument("csv_path", help="Path to the APD CSV file")
    parser.add_argument("outdir", help="Directory to write TTL files into")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of rows per TTL chunk (default: 10,000)",
    )

    args = parser.parse_args()

    main(args.csv_path, args.outdir, args.chunk_size)
