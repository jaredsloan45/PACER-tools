"""
This script has been included solely in an effort to make our RDF-graph-build process
transparent; it was copy-pasted directly from SCALES's private infrastructure repo, and
has not been tested here! Our assumption is that, because the raw data used for this
portion of our graph comes from a private dataset, nobody besides us will run this
script. If we're incorrect about this assumption, feel free to contact us at
engineering@scales-okn.org.
"""

import os
import json
import logging
import sys
import argparse
from pathlib import Path
import utils
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm
from rdflib import Graph, Namespace, Literal, RDF, XSD

sys.path.append(str(Path.cwd().parents[1].resolve()))
import utils
from constants import SCALES, J, NC, NIBRS, OCCS
from support import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.FileHandler("error.log")
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)

_make_party_uri_fulton = lambda charge_id: utils._make_party_uri(f'ga-fulton-{int(charge_id)}', 0)


def _create_graph() -> Graph:
    """Create a blank graph and bind standard prefixes."""
    g = Graph()
    g.bind("scales", SCALES)
    g.bind("j", J)
    g.bind("nc", NC)
    g.bind("nibrs", NIBRS)
    g.bind("occs", OCCS)
    g.bind("rdf", RDF)
    return g

def _make_metadata_graph():
    """Take care of a few Fulton-County-Jail-related triples we might want to use at some point."""
    g = _create_graph()
    facility_uri = utils._make_generic_uri('Facility', 'ga-fulton-county-jail')
    g.add((facility_uri, NC.FacilityName, 'Fulton County Jail'))
    g.add((facility_uri, NC.PhysicalAddress, '901 Rice St NW, Atlanta, GA 30318'))
    g.add((facility_uri, OCCS.FacilityFunctionCode, '11-13 11 33')) # "Detention Center" (see https://niem.github.io/model/5.0/occs/FacilityFunctionCodeSimpleType/#diagram)
    return g

def _build_docket_subgraph(
    g: Graph,
    ucid: str,
    hearings: List[Dict[str, Any]],
    case_uri,
):
    """Convert the list of hearings to a Register-of-Actions style sub-graph."""
    if not hearings:
        return

    table_uri = utils._make_generic_uri("DocketTable", f"{ucid}")
    g.add((case_uri, J.RegisterOfActions, table_uri))
    g.add((table_uri, RDF.type, J.RegisterOfActions))

    for idx, hearing in enumerate(hearings):
        entry_uri = utils._make_docket_uri(ucid, idx)
        g.add((table_uri, J.RegisterAction, entry_uri))
        g.add((entry_uri, RDF.type, J.RegisterAction))

        # Filing / event date
        h_date = hearing.get("hearing_date")
        if h_date:
            g.add(
                (
                    entry_uri,
                    J.RegisterActionDate,
                    Literal(utils._date_to_xsd(h_date), datatype=XSD.date),
                )
            )

        # Description text (type, result, etc.)
        parts = [hearing.get("hearing_type")]
        if hearing.get("result"):
            parts.append(f": {hearing['result']}")
        if hearing.get("result_type"):
            parts.append(f"({hearing['result_type']})")
        contents = " ".join([p for p in parts if p])
        if contents:
            g.add(
                (
                    entry_uri,
                    J.RegisterActionDescriptionText,
                    Literal(utils._escape_quotes(contents)),
                )
            )


def process_json_file(json_path: str):
    """Parse a single Fulton-county *charge* JSON and return a list(triples)."""
    try:
        with open(json_path, "r") as fh:
            data = json.load(fh)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error reading %s: %s", json_path, exc)
        return None

    g = _create_graph()

    charge_id = data.get("charge_id")
    charge_uri = utils._make_generic_uri("Charge", f"ga-fulton-01-{int(charge_id)}")

    g.add((charge_uri, RDF.type, J.Charge))
    desc = data.get("charge_offense_description")
    g.add((charge_uri, J.ChargeText, Literal(utils._escape_quotes(desc))))

    severity = data.get("severity")
    g.add((charge_uri, J.ChargeSeverityLevelCode, Literal(severity)))

    # Charge decision / status
    # decision = data.get("charge_decision") or {}
    # if decision.get("charge_decision"):
    #     g.add(
    #         (
    #             charge_uri,
    #             J.ChargeDispositionCategoryText,
    #             Literal(decision["charge_decision"]),
    #         )
    #     )
    # if decision.get("charge_status"):
    #     g.add(
    #         (
    #             charge_uri,
    #             NC.StatusDescriptionText,
    #             Literal(decision["charge_status"]),
    #         )
    #     )
    # if decision.get("file_date"):
    #     g.add(
    #         (
    #             charge_uri,
    #             NC.StartDate,
    #             Literal(
    #                 utils._date_to_xsd(decision["file_date"]),
    #                 datatype=XSD.date,
    #             ),
    #         )
    #     )
    # if decision.get("charge_decision_date"):
    #     g.add(
    #         (
    #             charge_uri,
    #             NC.EndDate,
    #             Literal(
    #                 utils._date_to_xsd(decision["charge_decision_date"]),
    #                 datatype=XSD.date,
    #             ),
    #         )
    #     )

    case_info = data.get("case")
    if case_info:
        case_nbr = case_info.get("case_nbr")
        ucid = f"ga-fulton-01-{case_nbr}"
        case_uri = utils._make_case_uri(ucid)

        g.add((charge_uri, J.ChargeFiledCase, case_uri))
        g.add((case_uri, NC.CaseDocketID, Literal(utils._escape_quotes(case_nbr))))
        g.add((case_uri, RDF.type, nc.CourtCase))
        g.add((case_uri, RDF.type, SCALES.CriminalCase))
        g.add((case_uri, NC.CaseGeneralCategoryText, Literal("criminal")))

        # Hearings / register of actions
        _build_docket_subgraph(g, ucid, case_info.get("hearings", []), case_uri)

    if data.get("bond_type"):
        g.add((charge_uri, J.BondType, Literal(data["bond_type"])))
    if data.get("bond_amount"):
        try:
            amt = float(data["bond_amount"])
        except (TypeError, ValueError):
            amt = data["bond_amount"]
        g.add((charge_uri, J.BondAmount, Literal(amt, datatype=XSD.float)))

    booking = data.get("booking")
    if booking:
        booking_uri = utils._make_generic_uri('Booking', f"ga-fulton-{int(booking['jailing_id'])}")
        g.add((booking_uri, J.BookingFacility, utils._make_generic_uri('Facility', 'ga-fulton-county-jail')))
        party_uri = _make_party_uri_fulton(charge_id) # we don't create this uri earlier because the booking dict is where the party info resides

        # Link charge/booking/party
        g.add((charge_uri, J.Booking, booking_uri))
        g.add((booking_uri, RDF.type, J.Booking))
        g.add((party_uri, J.PersonCharge, charge_uri))
        g.add((party_uri, RDF.type, J.BookingSubject))

        # g.add((party_uri, J.ParticipantRoleCategoryText, Literal("defendant"))) # commented this out because not all arrestees become defendants
        if booking.get("gender"):
            g.add((party_uri, J.PersonSexCode, Literal(booking["gender"])))
        if booking.get("race"):
            g.add((party_uri, NC.PersonRaceText, Literal(booking["race"])))

        if booking.get("booking_date"):
            g.add(
                (
                    booking_uri,
                    NC.StartDate,
                    Literal(
                        utils._date_to_xsd(booking["booking_date"]), datatype=XSD.date
                    ),
                )
            )
        if booking.get("release_date"):
            g.add(
                (
                    booking_uri,
                    NC.EndDate,
                    Literal(
                        utils._date_to_xsd(booking["release_date"]), datatype=XSD.date
                    ),
                )
            )

    return list(g)


def _write_graph_worker(graph: Graph, outdir: Path, file_name=None):
    utils._write_graph_to_file(graph, outdir, file_name=file_name)


def main(indir: str, outdir: str):
    """Read all JSON charge files beneath *indir* and emit Turtle files to *outdir*."""
    indir_p = Path(indir)
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    utils._write_graph_to_file(_make_metadata_graph(), outdir, file_name='facility.ttl')

    json_files = [str(f) for f in indir_p.rglob("*.json") if f.is_file()]
    logger.info("Discovered %d JSON files in %s", len(json_files), indir)

    record_counter = 0
    global_graph = _create_graph()
    write_futures = []
    
    with ProcessPoolExecutor(max_workers=12) as proc_exec, ThreadPoolExecutor(
        max_workers=8
    ) as thread_exec:
        futures = {proc_exec.submit(process_json_file, jf): jf for jf in json_files}

        with tqdm(total=len(json_files), desc="Processing charges") as pbar:
            for future in as_completed(futures):
                triples = future.result()
                if triples:
                    for triple in triples:
                        global_graph.add(triple)
                    record_counter += 1

                    # Flush every 10k records (adjust as needed)
                    if record_counter >= 10000:
                        wf = thread_exec.submit(
                            _write_graph_worker, global_graph, outdir_p
                        )
                        write_futures.append(wf)
                        record_counter = 0
                        global_graph = _create_graph()

                pbar.update(1)

    # Final flush
    if record_counter:
        utils._write_graph_to_file(global_graph, outdir_p)

    # Await parallel writers
    for wf in as_completed(write_futures):
        try:
            wf.result()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error in write operation: %s", exc)

    # entities (added by scott)
    utils.process_entities(
        (settings.PARTY_DIS_UNIVERSAL,),
        outdir,
        _make_party_uri_fulton,
        ('charge_id',),
        filter_funcs={settings.PARTY_DIS_UNIVERSAL: (
            lambda df: df[df.court.eq('ga-fulton')])
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Fulton-county charge JSON files and emit Turtle graphs",
    )
    parser.add_argument("indir", help="Directory containing input JSON files")
    parser.add_argument("outdir", help="Directory where TTL files will be written")
    args = parser.parse_args()

    main(args.indir, args.outdir)
