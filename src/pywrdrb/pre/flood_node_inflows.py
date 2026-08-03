"""
Preprocessors for generating catchment inflows at flood monitoring locations.

The three flood-monitoring nodes (Hale Eddy, Fishs Eddy, Bridgeville) sit between
an at-dam release gage and a downstream convergence node. The dataset's catchment
inflow file already contains their local runoff -- it is just placed too far
downstream, lumped into ``delLordville`` / ``delMontague``. These preprocessors
*relocate* that water to where it physically enters the river.

Design principle: STRICT MASS-CONSERVING REDISTRIBUTION.
    Each flood node's inflow is a fixed drainage-area fraction of its immediate
    downstream node's ALREADY-MARGINAL catchment inflow, and that same amount is
    subtracted from the downstream node. Fractions sum to well under 1, so the
    downstream residual can never go negative. Net basin flow -- and therefore
    flow at Montague and Trenton -- is unchanged to full float precision. No
    water is created, no bias correction is applied.

Provides two preprocessors that share the same per-trace math:
- FloodNodeInflowPreprocessor: single-trace, CSV in / CSV out.
- FloodNodeInflowEnsemblePreprocessor: ensemble, HDF5 in / HDF5 out, MPI-parallel.

Change Log:
TJA, 2026-01-08, Initial implementation for flood monitoring nodes.
TJA, 2026-01-13, Fixed mass balance by subtracting flood node flows from downstream marginal inflows.
TJA, 2026-05-06, Refactored math into pure helpers; added ensemble preprocessor.
TJA, 2026-07-31, Corrected drainage areas against USGS NWIS (01425000 515->456,
    01417000 705->372, 01436000 150->92.6, 01436690 160->171, delLordville
    1595->1590) and removed the erroneous double subtraction of upstream inflows:
    the input catchment_inflow_mgd.csv is already marginal (see
    pywrdrb.pre.flows._subtract_upstream_catchment_inflows), so subtracting the
    upstream nodes a second time drove the intermediate flow negative on 92.5% of
    days and reduced the flood-node inflows to ~2% of their physical magnitude.
    Bridgeville now scales from delMontague's marginal inflow rather than the
    at-dam gage 01436000, whose marginal catchment is ~0 sq mi.
    NOTE: any existing catchment_inflow_with_flood_nodes_mgd.{csv,hdf5} generated
    before this date is stale and must be regenerated (the ensemble preprocessor
    short-circuits on output existence unless force=True).
"""

import os
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pywrdrb.pre._mpi_utils import bcast_with_error, point_to_point_gather
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor


__all__ = [
    "DRAINAGE_AREAS",
    "DRAINAGE_AREA_SOURCES",
    "FLOOD_NODE_IDS",
    "validate_drainage_areas",
    "incremental_drainage_areas",
    "flood_node_inflow_fractions",
    "compute_hale_eddy_inflow",
    "compute_fishs_eddy_inflow",
    "compute_bridgeville_inflow",
    "subtract_flood_inflows_from_downstream",
    "add_flood_nodes_to_inflows",
    "FloodNodeInflowPreprocessor",
    "FloodNodeInflowEnsemblePreprocessor",
]


# ---------------------------------------------------------------------------
# Drainage areas (sq mi).
#
# Every value is sourced. The three at-dam release gages (01425000, 01417000,
# 01436000) sit immediately below their reservoirs, so their drainage areas are
# essentially identical to the reservoir catchments and their MARGINAL catchments
# are ~0 sq mi. That is physically correct and is checked by
# validate_drainage_areas(); a previous version of this table inflated them to
# 515 / 705 / 150, which fabricated 60 / 333 / 57.5 sq mi of nonexistent
# below-dam catchment and mis-sized every flood-node increment.
# ---------------------------------------------------------------------------
DRAINAGE_AREAS = {
    # Reservoir catchments
    "cannonsville": 455.0,
    "pepacton": 372.0,
    "neversink": 92.5,

    # At-dam release gages (marginal catchment ~0 sq mi)
    "01425000": 456.0,      # W Br Delaware R at Stilesville NY
    "01417000": 372.0,      # E Br Delaware R at Downsville NY
    "01436000": 92.6,       # Neversink R at Neversink NY

    # Flood monitoring nodes
    "01426500": 595.0,      # W Br Delaware R at Hale Eddy NY
    "01421000": 784.0,      # E Br Delaware R at Fishs Eddy NY
    "01436690": 171.0,      # Neversink R at Bridgeville NY

    # Downstream convergence points
    "delLordville": 1590.0,  # Delaware R at Lordville NY (01427207)
    "delMontague": 3480.0,   # Delaware R at Montague NJ (01438500)

    # Other tributaries entering between Lordville and Montague. Needed only to
    # size delMontague's marginal catchment (the donor for Bridgeville).
    "01433500": 200.0,       # Mongaup R near Mongaup NY
    "prompton": 40.6,        # W Br Lackawaxen R near Aldenville PA (01428750)
    "wallenpaupack": 219.0,  # Lake Wallenpaupack / Wallenpaupack Ck watershed
    "shoholaMarsh": 26.0,    # Shohola Marsh Reservoir (headwater of Shohola Ck)
}

_USGS = "USGS NWIS monitoring-location page, retrieved 2026-07-31"

DRAINAGE_AREA_SOURCES = {
    "cannonsville": "NYCDEP Cannonsville Reservoir watershed",
    "pepacton": "NYCDEP Pepacton Reservoir watershed",
    "neversink": "NYCDEP Neversink Reservoir watershed",
    "01425000": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01425000/",
    "01417000": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01417000/",
    "01436000": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01436000/",
    "01426500": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01426500/",
    "01421000": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01421000/",
    "01436690": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01436690/",
    "delLordville": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01427207/",
    "delMontague": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01438500/",
    "01433500": f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01433500/",
    "prompton": (
        f"{_USGS}: https://waterdata.usgs.gov/monitoring-location/01428750/ "
        "(the gage the prompton node is matched to in obs_site_matches)"
    ),
    "wallenpaupack": (
        "Lake Wallenpaupack Watershed Management District, "
        "https://wallenpaupackwatershed.org/about/lake-wallenpaupack/ "
        "(219 sq mi; approximate, no USGS gage at the outlet)"
    ),
    "shoholaMarsh": (
        "Approximate headwater area of Shohola Creek above Shohola Marsh "
        "Reservoir; no USGS gage. Shohola Ck at Shohola (01432512) is 84.7 sq mi "
        "far downstream. Only affects the delMontague marginal-area denominator; "
        "a 0->85 sq mi swing moves the Bridgeville increment by <5%."
    ),
}


# Flood-monitoring node IDs in the order they are added to the inflow frame.
FLOOD_NODE_IDS = ("01426500", "01421000", "01436690")


# Mapping from flood node to its immediate downstream node (used for mass balance
# and as the donor for that node's drainage-area redistribution).
_FLOOD_DOWNSTREAM_MAP = {
    "01426500": "delLordville",  # Hale Eddy -> Lordville
    "01421000": "delLordville",  # Fishs Eddy -> Lordville
    "01436690": "delMontague",   # Bridgeville -> Montague
}


# Nodes whose (already-marginal) inflow columns are subtracted from delMontague by
# pywrdrb.pre.flows._subtract_upstream_catchment_inflows. The sum of their marginal
# drainage areas is what must be removed from Montague's total to size Montague's
# own marginal catchment. cannonsville/pepacton/01425000/01417000 are inside
# delLordville's 1590 sq mi and so are already accounted for there.
_MONTAGUE_UPSTREAM_AREAS = (
    "delLordville",   # entire upper basin above Lordville
    "01436000",       # Neversink R at Neversink (incl. neversink reservoir)
    "01433500",       # Mongaup R near Mongaup (incl. mongaupeCombined)
    "prompton",
    "wallenpaupack",
    "shoholaMarsh",
)


def incremental_drainage_areas():
    """Local (unregulated) drainage area of each flood node, sq mi.

    Each flood node drains the reach between its upstream at-dam release gage and
    the gage itself.
    """
    return {
        "01426500": DRAINAGE_AREAS["01426500"] - DRAINAGE_AREAS["01425000"],
        "01421000": DRAINAGE_AREAS["01421000"] - DRAINAGE_AREAS["01417000"],
        "01436690": DRAINAGE_AREAS["01436690"] - DRAINAGE_AREAS["01436000"],
    }


def _marginal_drainage_areas():
    """Drainage area of each donor node's own marginal catchment, sq mi."""
    lordville = (
        DRAINAGE_AREAS["delLordville"]
        - DRAINAGE_AREAS["01425000"]
        - DRAINAGE_AREAS["01417000"]
    )
    montague = DRAINAGE_AREAS["delMontague"] - sum(
        DRAINAGE_AREAS[n] for n in _MONTAGUE_UPSTREAM_AREAS
    )
    return {"delLordville": lordville, "delMontague": montague}


def flood_node_inflow_fractions():
    """Fraction of each donor node's marginal inflow assigned to each flood node.

    This is the entire model: ``inflow[flood_node] = fraction * inflow[donor]``.
    Because the two Lordville fractions sum to well under 1 (and the Montague
    fraction is small), the donor residual is always non-negative -- redistribution
    can never create or destroy water.
    """
    inc = incremental_drainage_areas()
    marg = _marginal_drainage_areas()
    return {
        fid: inc[fid] / marg[_FLOOD_DOWNSTREAM_MAP[fid]] for fid in FLOOD_NODE_IDS
    }


def validate_drainage_areas(areas=None, sources=None):
    """Check the drainage-area table for internal and physical consistency.

    Raises ``ValueError`` on any failure. Called once at import so a bad edit
    fails immediately rather than silently producing wrong inflows.

    Checks
    ------
    1. At-dam consistency: each release gage's area is within 10% of its
       reservoir catchment. The pre-2026-07-31 table failed this at 13% / 90% /
       62%, which is the single check that catches that class of error.
    2. Positivity: every increment and every donor marginal area is > 0. In
       particular the Lordville residual (1590 - 595 - 784 = 211 sq mi) must be
       positive, i.e. the two flood gauges cannot span more than Lordville.
    3. Fractions: the fractions assigned from any one donor sum to < 1, which is
       what guarantees strict mass conservation.
    4. Provenance: every area has a non-empty source string.

    Note that "closure" -- increments plus residual equalling the donor's
    marginal area -- is an algebraic identity given these definitions, so it is
    not checked here (it can never fail). The positivity and at-dam checks are
    what actually constrain the table.
    """
    areas = DRAINAGE_AREAS if areas is None else areas
    sources = DRAINAGE_AREA_SOURCES if sources is None else sources
    errors = []

    def _inc(a, b):
        return areas[a] - areas[b]

    # 2. Positivity of every increment, donor marginal area, and the residual.
    inc_hale, inc_fishs = _inc("01426500", "01425000"), _inc("01421000", "01417000")
    inc_bridge = _inc("01436690", "01436000")
    marg_lordville = (
        areas["delLordville"] - areas["01425000"] - areas["01417000"]
    )
    marg_montague = areas["delMontague"] - sum(
        areas[n] for n in _MONTAGUE_UPSTREAM_AREAS
    )
    residual_lordville = (
        areas["delLordville"] - areas["01426500"] - areas["01421000"]
    )

    for name, value in [
        ("Hale Eddy increment", inc_hale),
        ("Fishs Eddy increment", inc_fishs),
        ("Bridgeville increment", inc_bridge),
        ("delLordville marginal", marg_lordville),
        ("delMontague marginal", marg_montague),
        ("Lordville residual", residual_lordville),
    ]:
        if value <= 0:
            errors.append(f"{name} must be > 0, got {value:.4g} sq mi")

    # 1. At-dam release gages must not imply a large below-dam catchment.
    for gage, reservoir in [
        ("01425000", "cannonsville"),
        ("01417000", "pepacton"),
        ("01436000", "neversink"),
    ]:
        rel = abs(areas[gage] - areas[reservoir]) / areas[reservoir]
        if rel > 0.10:
            errors.append(
                f"gage {gage} ({areas[gage]:g} sq mi) differs from its reservoir "
                f"catchment {reservoir} ({areas[reservoir]:g} sq mi) by "
                f"{100 * rel:.0f}% (>10%). These gages sit at the dam, so their "
                "areas should be nearly equal."
            )

    # 3. Fractions from a common donor must sum to < 1.
    if marg_lordville > 0 and marg_montague > 0:
        by_donor = {}
        for fid, inc in [
            ("01426500", inc_hale),
            ("01421000", inc_fishs),
            ("01436690", inc_bridge),
        ]:
            donor = _FLOOD_DOWNSTREAM_MAP[fid]
            marg = marg_lordville if donor == "delLordville" else marg_montague
            by_donor[donor] = by_donor.get(donor, 0.0) + inc / marg
        for donor, total in by_donor.items():
            if total >= 1.0:
                errors.append(
                    f"flood-node fractions drawn from {donor} sum to {total:.4f} "
                    ">= 1; redistribution would not be mass conserving"
                )

    # 4. Provenance.
    missing = sorted(set(areas) - {k for k, v in sources.items() if v})
    if missing:
        errors.append(f"drainage areas without a source: {missing}")

    if errors:
        raise ValueError(
            "invalid drainage-area table:\n  - " + "\n  - ".join(errors)
        )


validate_drainage_areas()


# ---------------------------------------------------------------------------
# Pure helpers (operate on a per-trace pd.DataFrame; no I/O, no logging).
# ---------------------------------------------------------------------------

def _redistribute_from_downstream(inflows: pd.DataFrame, flood_node: str) -> pd.Series:
    """Carve a flood node's local inflow out of its downstream node's marginal inflow.

    ``inflows`` must already be MARGINAL (per-node local catchment inflow), which
    is what ``catchment_inflow_mgd.csv`` contains -- ``pywrdrb.pre.flows``
    subtracts upstream contributions before writing it. The downstream node's
    marginal catchment physically contains the flood node's reach, so taking a
    drainage-area fraction of it moves that runoff to the right place without
    changing the total.

    Do NOT subtract upstream nodes here. Doing so double-counts a subtraction that
    ``_subtract_upstream_catchment_inflows`` has already performed and drives the
    result negative on the great majority of days.
    """
    donor = _FLOOD_DOWNSTREAM_MAP[flood_node]
    if donor not in inflows.columns:
        raise KeyError(
            f"cannot compute inflow for flood node {flood_node}: donor column "
            f"'{donor}' is missing from the inflow frame. Available columns: "
            f"{sorted(inflows.columns)}"
        )
    fraction = flood_node_inflow_fractions()[flood_node]
    # .clip is defensive only: marginal inflows are already non-negative.
    return (inflows[donor] * fraction).clip(lower=0)


def compute_hale_eddy_inflow(inflows: pd.DataFrame) -> pd.Series:
    """Local catchment inflow at Hale Eddy (01426500), MGD.

    The 139 sq mi of West Branch Delaware between Stilesville (01425000, at
    Cannonsville dam) and Hale Eddy, taken as a drainage-area share of
    ``delLordville``'s marginal inflow.
    """
    return _redistribute_from_downstream(inflows, "01426500")


def compute_fishs_eddy_inflow(inflows: pd.DataFrame) -> pd.Series:
    """Local catchment inflow at Fishs Eddy (01421000), MGD.

    The 412 sq mi of East Branch Delaware between Downsville (01417000, at
    Pepacton dam) and Fishs Eddy -- which includes the large unregulated Beaver
    Kill -- taken as a drainage-area share of ``delLordville``'s marginal inflow.
    This is by far the largest of the three increments; the previous table sized
    it at 79 sq mi.
    """
    return _redistribute_from_downstream(inflows, "01421000")


def compute_bridgeville_inflow(inflows: pd.DataFrame) -> pd.Series:
    """Local catchment inflow at Bridgeville (01436690), MGD.

    The 78.4 sq mi of Neversink River between the Neversink release gage
    (01436000, at the dam) and Bridgeville, taken as a drainage-area share of
    ``delMontague``'s marginal inflow.

    The donor is delMontague rather than 01436000 because 01436000 sits at the
    dam: its marginal catchment is ~0.1 sq mi and its marginal inflow averages
    ~5 MGD, so scaling from it produced a physically meaningless 0.35 MGD. The
    dataset places this reach's runoff in delMontague's marginal catchment, which
    is where it must be taken from.
    """
    return _redistribute_from_downstream(inflows, "01436690")


def subtract_flood_inflows_from_downstream(
    inflows: pd.DataFrame, warn: bool = True
) -> pd.DataFrame:
    """
    Subtract flood-node inflows from their downstream nodes' marginal inflows.

    Returns a modified copy. Flows that go negative after subtraction (which
    should not happen for a correctly-shaped intermediate flow) are floored
    at zero. When ``warn=True`` (default) a single ``UserWarning`` is raised
    summarizing how many timesteps got clamped, across all downstream nodes.
    """
    out = inflows.copy()
    n_negative_total = 0

    for flood_node, downstream_node in _FLOOD_DOWNSTREAM_MAP.items():
        if flood_node in out.columns and downstream_node in out.columns:
            out[downstream_node] = out[downstream_node] - out[flood_node]
            n_neg = int((out[downstream_node] < 0).sum())
            n_negative_total += n_neg
            if n_neg > 0:
                out.loc[out[downstream_node] < 0, downstream_node] = 0

    if warn and n_negative_total > 0:
        warnings.warn(
            f"{n_negative_total} downstream-node values went negative after "
            "flood-node subtraction; floored to zero.",
            UserWarning,
            stacklevel=2,
        )
    return out


def add_flood_nodes_to_inflows(inflows: pd.DataFrame) -> pd.DataFrame:
    """
    Add the three flood-monitoring node columns to ``inflows`` and rebalance
    ``delLordville`` / ``delMontague`` so per-timestep mass is preserved.

    Returns a new DataFrame. The input is not mutated.
    """
    out = inflows.copy()
    out["01426500"] = compute_hale_eddy_inflow(out)
    out["01421000"] = compute_fishs_eddy_inflow(out)
    out["01436690"] = compute_bridgeville_inflow(out)
    out = subtract_flood_inflows_from_downstream(out)
    return out


# ---------------------------------------------------------------------------
# Single-trace preprocessor (CSV in / CSV out).
# ---------------------------------------------------------------------------

class FloodNodeInflowPreprocessor(DataPreprocessor):
    """
    Generates inflow data for flood monitoring nodes using drainage area scaling.

    Saves to: catchment_inflow_with_flood_nodes_mgd.csv
    """

    def __init__(self, inflow_type="nhmv10"):
        super().__init__()
        self.inflow_type = inflow_type

        # Use the shortcut registry (pn.sc) rather than pn.flows.get_str so
        # externally-registered inflow types resolve correctly. pn.flows scans
        # the filesystem with follow_symlinks=False and won't see paths added
        # at runtime via pn.sc.add(...).
        flows_dir = str(self.pn.sc.get(f"flows/{inflow_type}"))
        self.input_dirs = {"catchment_inflow": flows_dir}
        self.output_dirs = {"augmented_inflow": flows_dir}

    def load(self):
        """Load original catchment inflow data."""
        inflow_file = (
            Path(self.input_dirs["catchment_inflow"]) / "catchment_inflow_mgd.csv"
        )
        if not inflow_file.exists():
            raise FileNotFoundError(f"Base inflow file not found: {inflow_file}")

        df = pd.read_csv(inflow_file, index_col=0, parse_dates=True)
        self.raw_data["inflows"] = df
        print(f"Loaded base inflows: {df.shape[0]} timesteps, {df.shape[1]} nodes")

    def process(self):
        """Calculate flood node inflows by mass-conserving drainage-area redistribution."""
        print("\n" + "=" * 68)
        print("Flood Node Inflow Preprocessor")
        print("=" * 68)

        original = self.raw_data["inflows"]
        inflows = add_flood_nodes_to_inflows(original)
        self.processed_data["augmented_inflows"] = inflows

        inc = incremental_drainage_areas()
        fractions = flood_node_inflow_fractions()

        # Unit-area yield and non-zero fraction are the two numbers that make an
        # implausible inflow series obvious at a glance. Humid temperate
        # catchments in this basin run ~1-2 MGD/sq mi.
        print("\nFlood node local catchment inflows:")
        header = f"  {'node':10s} {'DA':>7s} {'frac':>8s} {'mean':>9s} {'yield':>9s} {'days>0':>8s}"
        print(header)
        print(f"  {'':10s} {'sq mi':>7s} {'of donor':>8s} {'MGD':>9s} {'MGD/sq mi':>9s} {'':>8s}")
        for node in FLOOD_NODE_IDS:
            mean = inflows[node].mean()
            print(
                f"  {node:10s} {inc[node]:7.1f} {fractions[node]:8.4f} "
                f"{mean:9.1f} {mean / inc[node]:9.2f} "
                f"{(inflows[node] > 0).mean():7.1%}"
            )

        # Mass conservation is the design guarantee -- report it, don't assume it.
        print("\nMass balance (redistribution must not change net basin flow):")
        for donor in ("delLordville", "delMontague"):
            if donor not in original.columns:
                continue
            taken = sum(
                inflows[fid]
                for fid in FLOOD_NODE_IDS
                if _FLOOD_DOWNSTREAM_MAP[fid] == donor
            )
            residual = inflows[donor]
            err = float((original[donor] - residual - taken).abs().max())
            print(
                f"  {donor:14s} retains {residual.sum() / original[donor].sum():.6f} "
                f"of its inflow; max |orig - residual - moved| = {err:.3e} MGD"
            )
        total_before = float(original.to_numpy().sum())
        total_after = float(inflows.to_numpy().sum())
        print(
            f"  basin total    before = {total_before:.4f}, after = {total_after:.4f}, "
            f"difference = {total_after - total_before:+.6e} MGD"
        )

    def save(self):
        """Save augmented inflow file."""
        output_dir = Path(self.output_dirs["augmented_inflow"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "catchment_inflow_with_flood_nodes_mgd.csv"

        inflows = self.processed_data["augmented_inflows"]
        inflows.to_csv(output_file)

        print(f"\nSaved augmented inflows to: {output_file}")
        print(f"  Total nodes: {inflows.shape[1]}")
        print(f"  Timesteps: {inflows.shape[0]}")


# ---------------------------------------------------------------------------
# Ensemble preprocessor (HDF5 in / HDF5 out, MPI-parallel).
# ---------------------------------------------------------------------------

INPUT_HDF5_NAME = "catchment_inflow_mgd.hdf5"
OUTPUT_HDF5_NAME = "catchment_inflow_with_flood_nodes_mgd.hdf5"


class FloodNodeInflowEnsemblePreprocessor(DataPreprocessor):
    """
    Ensemble counterpart of FloodNodeInflowPreprocessor.

    Reads a node-first ensemble HDF5 (``catchment_inflow_mgd.hdf5``), applies
    drainage-area scaling per realization to add the three flood-monitoring
    nodes (and rebalance ``delLordville`` / ``delMontague``), and writes a new
    node-first HDF5 (``catchment_inflow_with_flood_nodes_mgd.hdf5``) next to
    the input. The augmented file is a self-contained replacement for the
    input — every original node group is rewritten (with delLordville and
    delMontague modified) and three flood-node groups are added.

    Parallelization mirrors PredictedInflowEnsemblePreprocessor:
    ``np.array_split`` distributes realizations across ranks; each rank reads
    its own slice from the HDF5 concurrently; results are merged onto rank 0
    via point-to-point gather and saved.

    Parameters
    ----------
    inflow_type : str
        Flow data source label (e.g., 'nhmv10').
    realization_ids : list of int or str, optional
        Realization IDs to process. If None, every realization in the input
        HDF5 is processed.
    use_mpi : bool, default False
        If True, use MPI for per-realization parallelism.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to ``MPI.COMM_WORLD`` when ``use_mpi=True``.
    force : bool, default False
        If True, rebuild the output even if it already exists.
    """

    def __init__(
        self,
        inflow_type="nhmv10",
        realization_ids=None,
        use_mpi=False,
        comm=None,
        force=False,
    ):
        super().__init__()
        self.inflow_type = inflow_type
        self.realization_ids = realization_ids
        self.force = force

        if use_mpi:
            if comm is None:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        self.use_mpi = use_mpi

        # Use the shortcut registry (pn.sc) rather than pn.flows.get_str so
        # externally-registered inflow types resolve correctly. pn.flows scans
        # the filesystem with follow_symlinks=False and won't see paths added
        # at runtime via pywrdrb.load_pn_config(...) or pn.sc.add(...). Every
        # other ensemble preprocessor (PredictedInflowEnsemblePreprocessor,
        # FlowEnsemble, PredictionEnsemble) uses pn.sc.get for the same reason.
        flows_dir = str(self.pn.sc.get(f"flows/{inflow_type}"))
        self.input_dirs = {INPUT_HDF5_NAME: flows_dir}
        self.output_dirs = {OUTPUT_HDF5_NAME: flows_dir}

        # Populated by load() / process().
        self._input_path = os.path.join(flows_dir, INPUT_HDF5_NAME)
        self._output_path = os.path.join(flows_dir, OUTPUT_HDF5_NAME)
        self._node_names = None
        self._dates = None
        self._my_realization_data = {}
        self._local_processed = {}

    # ---- I/O helpers --------------------------------------------------

    def _resolve_realization_ids(self):
        """Read realization IDs (as strings) from the input HDF5 on rank 0."""
        if self.realization_ids is not None:
            return [str(r) for r in self.realization_ids]
        with h5py.File(self._input_path, "r") as f:
            first_node = list(f.keys())[0]
            labels = f[first_node].attrs["column_labels"]
            return [str(label) for label in labels]

    def _resolve_node_names(self):
        """Read input HDF5 top-level group names on rank 0."""
        with h5py.File(self._input_path, "r") as f:
            return [k for k in f.keys() if isinstance(f[k], h5py.Group)]

    def _read_one_realization(self, hdf5_file, realization_id, node_names):
        """Build a per-realization DataFrame from an open HDF5 file handle."""
        data = {}
        for node in node_names:
            node_group = hdf5_file[node]
            data[node] = node_group[str(realization_id)][:]
        return data

    def _read_dates(self, hdf5_file, node_names):
        """Read the canonical date axis from the input HDF5."""
        # The single-trace input HDF5 uses 'date' (per existing
        # PredictedInflowEnsemblePreprocessor); fall back to 'datetime'
        # for forward compatibility.
        node_group = hdf5_file[node_names[0]]
        if "date" in node_group:
            raw = node_group["date"][:]
        elif "datetime" in node_group:
            raw = node_group["datetime"][:]
        else:
            raise KeyError(
                f"Neither 'date' nor 'datetime' dataset present in "
                f"/{node_names[0]} of {self._input_path}"
            )
        return [d.decode() if isinstance(d, bytes) else str(d) for d in raw]

    # ---- ABC interface -------------------------------------------------

    def load(self):
        """Resolve realization IDs and node names; each rank reads its slice."""
        if not os.path.exists(self._input_path):
            raise FileNotFoundError(
                f"Base ensemble inflow file not found: {self._input_path}"
            )

        if self.use_mpi:
            self.realization_ids = bcast_with_error(
                self.comm, self.rank, self._resolve_realization_ids
            )
            self._node_names = bcast_with_error(
                self.comm, self.rank, self._resolve_node_names
            )
        else:
            self.realization_ids = self._resolve_realization_ids()
            self._node_names = self._resolve_node_names()

        if self.use_mpi:
            my_ids = list(
                np.array_split(self.realization_ids, self.size)[self.rank]
            )
            self.comm.Barrier()
        else:
            my_ids = list(self.realization_ids)

        t0 = time.time()
        with h5py.File(self._input_path, "r") as f:
            for rid in my_ids:
                self._my_realization_data[str(rid)] = (
                    self._read_one_realization(f, rid, self._node_names)
                )
            # Read canonical date axis from one (any) node group on every rank
            # so save() on rank 0 always has it without an extra gather.
            self._dates = self._read_dates(f, self._node_names)

        if self.rank == 0:
            print(
                f"[rank {self.rank}/{self.size}] load: read "
                f"{len(my_ids)} realizations in {time.time() - t0:.1f}s "
                f"(of {len(self.realization_ids)} total)"
            )
        if self.use_mpi:
            self.comm.Barrier()

    def process(self):
        """Apply add_flood_nodes_to_inflows per assigned realization."""
        # Idempotency: if output already exists and force is False, skip.
        if not self.force and os.path.exists(self._output_path):
            if self.rank == 0:
                print(
                    f"Output already exists at {self._output_path}; "
                    "skipping (set force=True to rebuild)."
                )
            self._local_processed = {}
            self.processed_data["augmented_inflows"] = {}
            return

        if not self._my_realization_data:
            self.load()

        index = pd.to_datetime(pd.Index(self._dates))
        local = {}
        for rid, node_arrays in self._my_realization_data.items():
            df_in = pd.DataFrame(node_arrays, index=index)
            local[rid] = add_flood_nodes_to_inflows(df_in)
        self._local_processed = local

        if self.use_mpi:
            merged = point_to_point_gather(
                self.comm, self.rank, self.size, local
            )
            if self.rank == 0:
                self.processed_data["augmented_inflows"] = merged
            else:
                self.processed_data["augmented_inflows"] = {}
        else:
            self.processed_data["augmented_inflows"] = dict(local)

    def save(self):
        """Write the augmented HDF5 in node-first schema (rank 0 only)."""
        if self.use_mpi and self.rank != 0:
            self.comm.Barrier()
            return

        augmented = self.processed_data.get("augmented_inflows", {})
        if not augmented:
            # No-op: process() short-circuited (output already present).
            if self.use_mpi:
                self.comm.Barrier()
            return

        rid_order = [str(r) for r in self.realization_ids]
        missing = [r for r in rid_order if r not in augmented]
        if missing:
            raise RuntimeError(
                f"Augmented inflows missing realizations: {missing[:5]}..."
            )

        # Use the first realization's columns as the canonical column order.
        first_rid = rid_order[0]
        all_columns = list(augmented[first_rid].columns)

        # Sanity: every realization must share the same column set.
        for rid in rid_order[1:]:
            if list(augmented[rid].columns) != all_columns:
                raise RuntimeError(
                    f"Augmented column set for realization {rid} differs "
                    f"from realization {first_rid}."
                )

        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)

        # h5py's variable-length string dtype requires a numpy object array;
        # passing fixed-width unicode (e.g. <U10) raises "No conversion path".
        date_strings = np.asarray([str(d) for d in self._dates], dtype=object)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        column_labels = np.asarray(rid_order, dtype=object)

        with h5py.File(self._output_path, "w") as hf:
            for node in all_columns:
                node_group = hf.create_group(node)
                node_group.attrs.create(
                    "column_labels", column_labels, dtype=str_dtype
                )
                for rid in rid_order:
                    node_group.create_dataset(
                        rid,
                        data=augmented[rid][node].to_numpy(dtype=float),
                    )
                node_group.create_dataset(
                    "date", data=date_strings, dtype=str_dtype
                )

        print(
            f"Saved augmented ensemble inflows to {self._output_path} "
            f"({len(all_columns)} nodes x {len(rid_order)} realizations x "
            f"{len(date_strings)} timesteps)"
        )
        if self.use_mpi:
            self.comm.Barrier()


def main():
    """Command-line interface for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate catchment inflows for flood monitoring nodes"
    )
    parser.add_argument(
        "--inflow-type",
        default="nhmv10",
        help=(
            "Flow data source, e.g. nhmv10, nwmv21, pub_nhmv10_BC_withObsScaled, "
            "or any inflow type registered at runtime via pywrdrb.load_pn_config"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the augmented file already exists",
    )

    args = parser.parse_args()

    # Resolve through the shortcut registry so runtime-registered inflow types
    # work; a free-form --inflow-type otherwise fails deep inside load().
    preprocessor = FloodNodeInflowPreprocessor(inflow_type=args.inflow_type)
    output_file = (
        Path(preprocessor.output_dirs["augmented_inflow"])
        / "catchment_inflow_with_flood_nodes_mgd.csv"
    )
    if output_file.exists() and not args.force:
        print(
            f"Output already exists at {output_file}; "
            "skipping (pass --force to rebuild)."
        )
        return

    preprocessor.load()
    preprocessor.process()
    preprocessor.save()

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
