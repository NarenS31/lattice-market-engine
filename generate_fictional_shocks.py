"""
generate_fictional_shocks.py — build the 2x2 shock catalogue (realism x framing).

The existing catalogue in supply_chain.py contains ten *real-event-adjacent*
supply chain shocks. Testing whether LLM agents respond to the economic
structure of a shock — rather than recalling how the market actually moved —
requires fictional shocks that are structurally identical but describe events
that never happened.

Design rules for the fictional counterparts
------------------------------------------
1. Same causal skeleton. Each fictional narrative is a slot-for-slot rewrite of
   its real counterpart: same clause order, same connectives, same second-order
   effect chain. Only the entities change.

2. Same numeric magnitudes, exactly. 18% stays 18%, 8-12 weeks stays 8-12
   weeks, 12% stays 12%, severity is copied. Matching rather than merely
   sampling from the same range removes magnitude as a confound entirely.

3. Invented entities only. Commodities, facilities, places, and firm types are
   coined for this file. None is a renaming of a specific documented event —
   the point is to defeat memorisation, so a fictional shock that is
   recognisably (say) the 2021 Suez blockage with the names filed off would
   reintroduce exactly the recall we are trying to exclude.

   Concretely, the fictional set avoids: named real firms, real place names,
   real commodities, and any event whose date/actors a model could pin down.
   The real set deliberately keeps them.

4. Sector counts match (three per event), and sector labels are invented at the
   same level of abstraction as the real ones.

The numeric framing — and its limitation
----------------------------------------
The numeric condition renders each event as a parameter vector (supply delta,
VIX, sentiment, fed rate, severity) instead of prose. Realism in that condition
survives *only* in the entity token (`lithium_carbonate` vs `seldrite_carbonate`);
every number is identical across the realism pair by construction.

That is deliberate — it is the tightest possible structural match — but it means
the realism manipulation is far weaker under numeric framing than under
narrative framing. Interpret a realism x framing interaction with that in mind:
it may reflect manipulation strength rather than a genuine moderating effect.
See the note printed by check_shock_balance.py.

Output
------
  python3 generate_fictional_shocks.py        # writes shocks.json (40 shocks)
  python3 generate_fictional_shocks.py --stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from supply_chain import SUPPLY_CHAIN_EVENTS

CATEGORY = "supply_chain_disruption"
OUTPUT_PATH = "shocks.json"

# Maps each real event key to its fictional counterpart key. Both members of a
# pair share a matched_pair_id in the output.
PAIRING = {
    "lithium_shortage": "seldrite_shortage",
    "semiconductor_shortage": "lattice_substrate_shortage",
    "energy_shock": "seyn_shock",
    "credit_crunch": "syndicate_crunch",
    # Families 5-10 (n=4 -> n=10). Same design rules; see the note above
    # FICTIONAL_EVENTS_EXT for the syllable-matching constraint that governs
    # how the invented names were chosen.
    "port_closure": "torvan_closure",
    "wheat_export_ban": "korr_export_ban",
    "pipeline_outage": "meridian_pipeline_outage",
    "flood_inundation": "saronde_inundation",
    "rail_embargo": "kesh_rail_embargo",
    "dam_failure": "korvath_dam_failure",
}


# ---------------------------------------------------------------------------
# Fictional catalogue — slot-for-slot counterparts of SUPPLY_CHAIN_EVENTS
# ---------------------------------------------------------------------------

FICTIONAL_EVENTS: dict[str, dict] = {
    # ↔ lithium_shortage (severity 0.7, 18%, extraction halt)
    "seldrite_shortage": {
        "affected_sectors": ["QV manufacturers", "module suppliers", "grid hardware"],
        "severity": 0.7,
        "narrative": (
            "A wildcat strike at the Karavelo seldrite mine has halted 18% of global "
            "seldrite carbonate supply, forcing QV manufacturers to cut production guidance "
            "and module suppliers to warn of margin compression."
        ),
        "multi_sentence_narrative": (
            "A wildcat strike at the Karavelo seldrite mine has halted 18% of global "
            "seldrite carbonate supply, forcing QV manufacturers to cut production guidance "
            "and module suppliers to warn of margin compression. "
            "This is expected to cascade into reduced QV sales forecasts and potential "
            "credit rating downgrades for affected manufacturers."
        ),
    },
    # ↔ semiconductor_shortage (severity 0.6, 8-12 weeks, facility fire)
    "lattice_substrate_shortage": {
        "affected_sectors": ["drivetrain manufacturers", "consumer devices", "avionics contractors"],
        "severity": 0.6,
        "narrative": (
            "A fire at an Ostrek fab in Torreval has disrupted K3 lattice production, with lead "
            "times extending 8–12 weeks across drivetrain and consumer device supply chains."
        ),
        "multi_sentence_narrative": (
            "A fire at an Ostrek fab in Torreval has disrupted K3 lattice production, with lead "
            "times extending 8–12 weeks across drivetrain and consumer device supply chains. "
            "Extended lead times are expected to stall drivetrain production schedules, delay consumer "
            "device product launches, and widen cost overruns for avionics contractors "
            "operating on fixed-price contracts."
        ),
    },
    # ↔ energy_shock (severity 0.8, 12%, chokepoint disruption)
    "seyn_shock": {
        "affected_sectors": ["industrials", "carriers", "chemical manufacturers"],
        "severity": 0.8,
        "narrative": (
            "A geopolitical disruption in the Strait of Vantry has spiked seyn futures 12%, "
            "triggering margin warnings across energy-intensive industries and forcing carrier "
            "capacity cuts."
        ),
        "multi_sentence_narrative": (
            "A geopolitical disruption in the Strait of Vantry has spiked seyn futures 12%, "
            "triggering margin warnings across energy-intensive industries and forcing carrier "
            "capacity cuts. "
            "Prolonged elevated energy costs are expected to accelerate pass-through inflation "
            "in consumer goods, trigger a wave of carrier route suspensions, and compress "
            "operating margins across chemical and industrial sectors through higher feedstock "
            "prices."
        ),
    },
    # ↔ credit_crunch (severity 0.9, three institutions, credit tightening)
    "syndicate_crunch": {
        "affected_sectors": ["clearing sector", "property development", "small caps"],
        "severity": 0.9,
        "narrative": (
            "Three mid-tier clearing houses have disclosed elevated syndicate loan exposure, "
            "triggering a broad credit tightening as counterparties reassess interbank "
            "lending terms."
        ),
        "multi_sentence_narrative": (
            "Three mid-tier clearing houses have disclosed elevated syndicate loan exposure, "
            "triggering a broad credit tightening as counterparties reassess interbank "
            "lending terms. "
            "Tighter credit conditions are expected to increase funding costs for property "
            "unit holders, restrict small-cap access to revolving credit facilities, and reduce "
            "overall market liquidity as banks shift toward shorter-duration and "
            "higher-grade collateral."
        ),
    },
}


# ---------------------------------------------------------------------------
# Families 5-10 — same four design rules, one added constraint
# ---------------------------------------------------------------------------
#
# Rule 1 (same causal skeleton) forces the invented names into the exact slots
# the real names occupy, which makes SYLLABLE COUNT the binding constraint here
# in a way it was not for the first four families. Flesch penalises syllables
# per word at 84.6x, so on a ~60-word narrative a fictional twin carrying just
# four extra syllables lands a ~5.6-point Flesch gap — over check_shock_balance's
# tolerance of 5, and it would show up as realism confounded with reading
# difficulty. Invented names are therefore syllable-matched to the real token
# they replace (Meishan/Meskan 2-2, Ningbo-Zhoushan/Torvan-Keshal 4-4), not
# merely invented.
#
# The opposite failure also had to be designed around: a name that is
# syllable-matched by being a near-homophone of the real one ("Nambo-Zhelan")
# would leak the source event and reintroduce exactly the recall rule 3 exists
# to defeat. Each name below is phonetically unrelated to its counterpart.
#
# One further trap, worth recording because it is invisible until the balance
# check runs: check_shock_balance's number regex reads "COVID-19" as the numeric
# literal -19, so a real narrative naming COVID-19 would fail the
# magnitudes-match test against a fictional twin containing no such number. The
# real text says "COVID" for that reason — still nameable and datable, no digit.

FICTIONAL_EVENTS_EXT: dict[str, dict] = {
    # ↔ port_closure (severity 0.65, 28%, terminal closure)
    "torvan_closure": {
        "affected_sectors": ["houseware importers", "module assemblers", "haulage brokers"],
        "severity": 0.65,
        "narrative": (
            "A Tessin outbreak at the Meskan terminal of the Port of Torvan-Keshal "
            "has idled 28% of the port's container throughput, forcing houseware "
            "importers to rebook freight and module assemblers to warn of delivery slippage."
        ),
        "multi_sentence_narrative": (
            "A Tessin outbreak at the Meskan terminal of the Port of Torvan-Keshal "
            "has idled 28% of the port's container throughput, forcing houseware "
            "importers to rebook freight and module assemblers to warn of delivery slippage. "
            "Idled berth capacity is expected to lengthen ocean transit times for houseware "
            "importers, push module assemblers into costly air freight substitution, "
            "and inflate demurrage charges for haulage brokers operating on fixed-rate contracts."
        ),
    },
    # ↔ wheat_export_ban (severity 0.55, 10 million tonnes, export ban)
    # Grain "korr" is one syllable to match "wheat", and the polity is "Ondar"
    # (two) to match "India" — the numeric token indian_wheat is only three
    # syllables, so the earlier ondaran_korrin overshot it by two and cost 10.6
    # Flesch points across a 16-word vector.
    "korr_export_ban": {
        "affected_sectors": ["meal processors", "herd operators", "commodity brokers"],
        "severity": 0.55,
        "narrative": (
            "Ondar's ban on korr exports has withdrawn 10 million tonnes from global "
            "grain supply, forcing meal processors to reprice contracts and herd "
            "operators to warn of feed cost inflation."
        ),
        "multi_sentence_narrative": (
            "Ondar's ban on korr exports has withdrawn 10 million tonnes from global "
            "grain supply, forcing meal processors to reprice contracts and herd "
            "operators to warn of feed cost inflation. "
            "Restricted export volumes are expected to raise input costs for meal processors, "
            "compress margins for herd operators, and widen basis risk for commodity "
            "brokers holding forward positions."
        ),
    },
    # ↔ pipeline_outage (severity 0.75, 2.5 million barrels/day, pipeline halt)
    # "elastomeric"/"conveyance" are syllable-matched to "petrochemical"/"logistics";
    # the shorter "polymer"/"haulage" left the fictional twin 6 syllables lighter
    # and 8.5 Flesch points easier, which is a readability confound, not a design.
    "meridian_pipeline_outage": {
        "affected_sectors": ["fuel wholesalers", "elastomeric refiners", "conveyance operators"],
        "severity": 0.75,
        "narrative": (
            "A ransomware attack on the Meridian Pipeline has halted 2.5 million barrels "
            "per day of refined fuel flow, forcing fuel wholesalers to invoke allocation "
            "and elastomeric refiners to cut run rates."
        ),
        "multi_sentence_narrative": (
            "A ransomware attack on the Meridian Pipeline has halted 2.5 million barrels "
            "per day of refined fuel flow, forcing fuel wholesalers to invoke allocation "
            "and elastomeric refiners to cut run rates. "
            "Suspended pipeline flow is expected to strand product inventories for fuel "
            "wholesalers, force run-rate cuts at elastomeric refiners, and escalate spot "
            "haulage rates for conveyance operators rerouting by tanker truck."
        ),
    },
    # ↔ flood_inundation (severity 0.7, 30%, estate flooding)
    "saronde_inundation": {
        "affected_sectors": ["QV manufacturers", "archive vendors", "contract fabricators"],
        "severity": 0.7,
        "narrative": (
            "Monsoon flooding across Saronde's Belhavet industrial estates has submerged "
            "plants producing 30% of global veldt drive output, forcing QV manufacturers to "
            "ration allocations and archive vendors to suspend price guidance."
        ),
        "multi_sentence_narrative": (
            "Monsoon flooding across Saronde's Belhavet industrial estates has submerged "
            "plants producing 30% of global veldt drive output, forcing QV manufacturers to "
            "ration allocations and archive vendors to suspend price guidance. "
            "Submerged production capacity is expected to delay build schedules for QV "
            "manufacturers, extend order backlogs for archive vendors, and idle assembly "
            "lines at contract fabricators dependent on single-source components."
        ),
    },
    # ↔ rail_embargo (severity 0.6, 15%, carload embargo)
    # The polity is "Kesh", not a two-syllable name: the real slot holds "US"
    # (one syllable), and the numeric entity token us_rail_carloads has only
    # 15 words to dilute a mismatch, so one extra syllable there is worth 5.6
    # Flesch points on its own.
    "kesh_rail_embargo": {
        "affected_sectors": ["bulk shippers", "nutrilizer producers", "rotor parts suppliers"],
        "severity": 0.6,
        "narrative": (
            "A labor dispute across Kesh Class I freight railroads has triggered an embargo "
            "cutting 15% of weekly carload volume, forcing bulk shippers to divert to truck "
            "and nutrilizer producers to halt outbound loading."
        ),
        "multi_sentence_narrative": (
            "A labor dispute across Kesh Class I freight railroads has triggered an embargo "
            "cutting 15% of weekly carload volume, forcing bulk shippers to divert to truck "
            "and nutrilizer producers to halt outbound loading. "
            "Reduced carload capacity is expected to strand harvest volumes for bulk shippers, "
            "force production curtailment at nutrilizer producers, and interrupt just-in-time "
            "deliveries for rotor parts suppliers running minimal buffer stock."
        ),
    },
    # ↔ dam_failure (severity 0.85, 40 million tonnes, tailings collapse)
    "korvath_dam_failure": {
        "affected_sectors": ["alloy producers", "ore desks", "extraction equipment suppliers"],
        "severity": 0.85,
        "narrative": (
            "A tailings dam collapse at Korvath's Serra do Dalmine mine has removed 40 million "
            "tonnes of annual ferrite ore capacity, forcing alloy producers to secure alternate "
            "offtake and ore desks to reprice term contracts."
        ),
        "multi_sentence_narrative": (
            "A tailings dam collapse at Korvath's Serra do Dalmine mine has removed 40 million "
            "tonnes of annual ferrite ore capacity, forcing alloy producers to secure alternate "
            "offtake and ore desks to reprice term contracts. "
            "Lost ore capacity is expected to raise input costs for alloy producers, tighten "
            "seaborne availability for ore desks, and defer capital orders at extraction "
            "equipment suppliers exposed to the affected operator."
        ),
    },
}

FICTIONAL_EVENTS.update(FICTIONAL_EVENTS_EXT)


# ---------------------------------------------------------------------------
# Numeric framing
# ---------------------------------------------------------------------------

# The entity token that carries realism in the numeric condition, and the
# magnitude token that must match across the realism pair.
NUMERIC_ENTITY = {
    "lithium_shortage":            ("lithium_carbonate",  "global_supply_delta", "-18.0%"),
    "seldrite_shortage":           ("seldrite_carbonate", "global_supply_delta", "-18.0%"),
    "semiconductor_shortage":      ("3nm_wafer",          "lead_time_delta",     "+8_to_12_weeks"),
    "lattice_substrate_shortage":  ("K3_lattice",         "lead_time_delta",     "+8_to_12_weeks"),
    "energy_shock":                ("crude_futures",      "front_month_delta",   "+12.0%"),
    "seyn_shock":               ("seyn_futures",    "front_month_delta",   "+12.0%"),
    "credit_crunch":               ("interbank_credit",   "counterparty_delta",  "-3_institutions"),
    "syndicate_crunch":            ("syndicate_credit",   "counterparty_delta",  "-3_institutions"),
    # Families 5-10. As with the first four, the delta field and value are
    # identical within each pair — realism is carried by the entity token alone,
    # which is the manipulation-strength caveat documented at the top of this file.
    "port_closure":            ("ningbo_container_slots", "throughput_delta",    "-28.0%"),
    "torvan_closure":          ("torvan_container_slots", "throughput_delta",    "-28.0%"),
    "wheat_export_ban":        ("indian_wheat",           "export_volume_delta", "-10.0_mmt"),
    "korr_export_ban":       ("ondar_korr",             "export_volume_delta", "-10.0_mmt"),
    "pipeline_outage":         ("colonial_refined_fuel",  "pipeline_flow_delta", "-2.5_mbd"),
    "meridian_pipeline_outage": ("meridian_refined_fuel",  "pipeline_flow_delta", "-2.5_mbd"),
    "flood_inundation":        ("hard_drive_output",      "output_delta",        "-30.0%"),
    "saronde_inundation":      ("veldt_drive_output",     "output_delta",        "-30.0%"),
    "rail_embargo":            ("us_rail_carloads",       "carload_delta",       "-15.0%"),
    "kesh_rail_embargo":     ("kesh_rail_carloads",     "carload_delta",       "-15.0%"),
    "dam_failure":             ("iron_ore_capacity",      "ore_capacity_delta",  "-40.0_mtpa"),
    "korvath_dam_failure":     ("ferrite_ore_capacity",   "ore_capacity_delta",  "-40.0_mtpa"),
}

EVENT_CLASS = {
    "lithium_shortage": "extraction_halt",
    "seldrite_shortage": "extraction_halt",
    "semiconductor_shortage": "fabrication_outage",
    "lattice_substrate_shortage": "fabrication_outage",
    "energy_shock": "transit_disruption",
    "seyn_shock": "transit_disruption",
    "credit_crunch": "credit_tightening",
    "syndicate_crunch": "credit_tightening",
    # Six new causal classes, none a re-sampling of the four above.
    "port_closure": "port_closure",
    "torvan_closure": "port_closure",
    "wheat_export_ban": "export_restriction",
    "korr_export_ban": "export_restriction",
    "pipeline_outage": "pipeline_outage",
    "meridian_pipeline_outage": "pipeline_outage",
    "flood_inundation": "flood_inundation",
    "saronde_inundation": "flood_inundation",
    "rail_embargo": "rail_embargo",
    "kesh_rail_embargo": "rail_embargo",
    "dam_failure": "dam_failure",
    "korvath_dam_failure": "dam_failure",
}

BASELINE_FED_RATE = 5.25


def derive_macro(severity: float) -> dict:
    """Map severity to the VIX / sentiment scale used by the legacy macro layer.

    Deterministic so a realism pair (identical severity) yields identical
    numbers — the entity token is the only thing that differs.
    """
    return {
        "vix": round(18.0 + severity * 20.0, 1),
        "sentiment": round(0.1 - severity * 0.9, 2),
        "fed_rate": BASELINE_FED_RATE,
    }


def numeric_text(event_key: str, severity: float) -> str:
    """Render an event as a flat parameter vector."""
    entity, delta_field, delta_value = NUMERIC_ENTITY[event_key]
    m = derive_macro(severity)
    return (
        f"event_class: {EVENT_CLASS[event_key]} | "
        f"entity: {entity} | "
        f"{delta_field}: {delta_value} | "
        f"vix: {m['vix']} | "
        f"sentiment: {m['sentiment']} | "
        f"fed_rate: {m['fed_rate']}% | "
        f"severity: {severity}"
    )


# ---------------------------------------------------------------------------
# Catalogue assembly
# ---------------------------------------------------------------------------

def _record(
    event_key: str,
    event: dict,
    realism: str,
    framing: str,
    pair_id: str,
) -> dict:
    if framing == "narrative":
        text = event["multi_sentence_narrative"]
    elif framing == "numeric":
        text = numeric_text(event_key, event["severity"])
    else:
        raise ValueError(f"unknown framing {framing!r}")

    return {
        "shock_id": f"{event_key}__{realism}__{framing}",
        "category": CATEGORY,
        "realism": realism,
        "framing": framing,
        "text": text,
        "matched_pair_id": pair_id,
        # --- supporting metadata (not part of the required schema) ---
        "event_key": event_key,
        "event_class": EVENT_CLASS[event_key],
        "severity": event["severity"],
        "affected_sectors": list(event["affected_sectors"]),
        "short_narrative": event["narrative"],
        "macro_params": derive_macro(event["severity"]),
    }


def build_catalogue() -> dict:
    """Return the full 2x2 catalogue: 4 events x 2 realism x 2 framing = 16."""
    shocks: list[dict] = []

    for real_key, fict_key in PAIRING.items():
        if real_key not in SUPPLY_CHAIN_EVENTS:
            raise KeyError(f"real event {real_key!r} missing from supply_chain.py")
        if fict_key not in FICTIONAL_EVENTS:
            raise KeyError(f"fictional event {fict_key!r} missing from this file")

        real_ev = SUPPLY_CHAIN_EVENTS[real_key]
        fict_ev = FICTIONAL_EVENTS[fict_key]

        if real_ev["severity"] != fict_ev["severity"]:
            raise ValueError(
                f"severity mismatch in pair {real_key}/{fict_key}: "
                f"{real_ev['severity']} vs {fict_ev['severity']}"
            )
        if len(real_ev["affected_sectors"]) != len(fict_ev["affected_sectors"]):
            raise ValueError(f"sector count mismatch in pair {real_key}/{fict_key}")

        pair_id = f"pair__{real_key}"
        for framing in ("narrative", "numeric"):
            shocks.append(_record(real_key, real_ev, "real", framing, pair_id))
            shocks.append(_record(fict_key, fict_ev, "fictional", framing, pair_id))

    return {
        "schema_version": 1,
        "category": CATEGORY,
        "design": {
            "factors": {
                "realism": ["real", "fictional"],
                "framing": ["narrative", "numeric"],
            },
            "n_event_families": len(PAIRING),
            "n_shocks": len(shocks),
            "cells": f"{len(PAIRING)} event families x 2 realism x 2 framing",
            "matching": (
                "Members of a matched_pair_id share causal structure, numeric "
                "magnitudes, severity, and sector count; they differ only in "
                "whether the entities are real or invented."
            ),
            "caveat": (
                "Under numeric framing, realism is carried solely by the entity "
                "token — every numeric value is identical within a pair. The "
                "realism manipulation is therefore weaker in the numeric cells "
                "than in the narrative cells."
            ),
        },
        "pairing": dict(PAIRING),
        "shocks": shocks,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the 2x2 shock catalogue.")
    ap.add_argument("-o", "--out", default=OUTPUT_PATH,
                    help=f"Output path (default {OUTPUT_PATH}).")
    ap.add_argument("--stdout", action="store_true",
                    help="Print the catalogue instead of writing it.")
    args = ap.parse_args(argv)

    catalogue = build_catalogue()
    payload = json.dumps(catalogue, indent=2, ensure_ascii=False)

    if args.stdout:
        print(payload)
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(payload + "\n")

    n = len(catalogue["shocks"])
    print(f"Wrote {n} shocks to {args.out}")
    for realism in ("real", "fictional"):
        for framing in ("narrative", "numeric"):
            c = sum(
                1 for s in catalogue["shocks"]
                if s["realism"] == realism and s["framing"] == framing
            )
            print(f"  {realism:<10} x {framing:<10} : {c}")
    print("\nNext: python3 check_shock_balance.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
