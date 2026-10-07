# -*- coding: cp1252 -*-
"""
Generic event-duration-class decomposition, shared between the persistent
(rolling-mean) compound-drought pipeline in fig_persistent.py and the
classic daily-coincidence pipeline in fig1.py.

Given any boolean/0-1 (time, lat, lon) "compound low-production day" field --
however it was derived (raw daily coincidence, or a rolling-mean-smoothed
"low week") -- this module walks contiguous runs of True into an event
table, then lets callers recompute annual frequency/duration/severity
restricted to events of a given duration class (exactly 1 day, exactly 2
days, 3+ days, ...). This is what powers the value-by-alpha decomposition
figures that ask "which event-duration class is driving the change".
"""
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Gap-tolerant event detection (event table)
# =============================================================================

def compute_event_table(da, time_dim="time"):
    """
    Long-format table with one row per (event day, lat, lon) for every
    contiguous run of True in boolean/0-1 DataArray `da`. Each row carries the
    event's total length in the 'duration' column (via a groupby transform),
    so the same table can be used both to compute annual event statistics and
    to rebuild a day-level mask of "events lasting >= N days" without
    re-walking the runs.
    """
    da = da.astype(int)
    first_time = pd.Timestamp(da[time_dim][0].values)
    da_pad = xr.concat(
        [xr.zeros_like(da.isel({time_dim: 0})).expand_dims(
             {time_dim: [first_time - pd.Timedelta(days=1)]}),
         da],
        dim=time_dim,
    )
    start_event = da_pad.diff(dim=time_dim, label="lower") > 0
    start_event[time_dim] = da[time_dim]
    id_event = start_event.cumsum(dim=time_dim) * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=("lat", "lon", time_dim)).dropna("z")
    df = pd.DataFrame({
        "event_id": stacked.values.astype(int),
        "lat":      stacked["lat"].values,
        "lon":      stacked["lon"].values,
        "time":     stacked[time_dim].values,
    })
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["year"] = df.groupby(["event_id", "lat", "lon"])["year"].transform("min")
    df["duration"] = df.groupby(["event_id", "lat", "lon"])["event_id"].transform("count")
    return df


def build_duration_class_mask(df, template_da, duration_class, time_dim="time"):
    """
    Boolean (time, lat, lon) mask, same shape as `template_da`, reconstructed
    from the event table: True on days belonging to an event whose total
    duration matches `duration_class`, a (op, value) pair with op in
    {'eq', 'ge'} (e.g. ('eq', 1) for single-day events, ('ge', 3) for events
    lasting 3 days or more). Vectorized via index lookup rather than
    re-walking the runs.
    """
    op, val = duration_class
    if op == "eq":
        sub = df[df["duration"] == val]
    elif op == "ge":
        sub = df[df["duration"] >= val]
    else:
        raise ValueError(f"Unknown duration_class op: {op!r}")

    times = template_da[time_dim].values
    lats = template_da.lat.values
    lons = template_da.lon.values
    mask = np.zeros((len(times), len(lats), len(lons)), dtype=bool)
    if not sub.empty:
        t_idx = pd.Index(times).get_indexer(sub["time"].values)
        lat_idx = pd.Index(lats).get_indexer(sub["lat"].values)
        lon_idx = pd.Index(lons).get_indexer(sub["lon"].values)
        mask[t_idx, lat_idx, lon_idx] = True
    return xr.DataArray(mask, dims=(time_dim, "lat", "lon"),
                         coords={time_dim: times, "lat": lats, "lon": lons})


def compute_annual_stats_for_duration_class(df_events_dedup, template_da, duration_class,
                                             time_dim="time"):
    """
    Annual (year, lat, lon) event frequency (count) and mean duration for
    events matching `duration_class` (see build_duration_class_mask for the
    (op, value) convention). Reindexed onto template_da's full grid/year
    range, 0-filled where no qualifying event occurred that year.
    """
    op, val = duration_class
    if op == "eq":
        sub = df_events_dedup[df_events_dedup["duration"] == val]
    elif op == "ge":
        sub = df_events_dedup[df_events_dedup["duration"] >= val]
    else:
        raise ValueError(f"Unknown duration_class op: {op!r}")

    full_years = np.unique(pd.DatetimeIndex(template_da[time_dim].values).year)
    freq = (sub.groupby(["year", "lat", "lon"]).size()
            .rename("frequency").to_xarray())
    dur = (sub.groupby(["year", "lat", "lon"])["duration"].mean()
           .rename("duration").to_xarray())
    freq = freq.reindex(year=full_years, lat=template_da.lat, lon=template_da.lon,
                         fill_value=0).fillna(0)
    dur = dur.reindex(year=full_years, lat=template_da.lat, lon=template_da.lon,
                       fill_value=0).fillna(0)
    return freq, dur


# Event-duration classes used by the value-by-alpha decomposition: label ->
# (op, value) as understood by build_duration_class_mask /
# compute_annual_stats_for_duration_class. "all" == ("ge", 1) since every
# event lasts at least 1 day by construction, so it is mathematically the
# unrestricted index -- kept as a class like any other rather than special-cased.
DECOMPOSITION_DURATION_CLASSES = [
    ("all", ("ge", 1)),
    ("eq1", ("eq", 1)),
    ("eq2", ("eq", 2)),
    ("ge3", ("ge", 3)),
]
DECOMPOSITION_CLASS_LABELS = ["All events", "Exactly 1 day", "Exactly 2 days", "3+ days"]


# =============================================================================
# Duration-class decomposition of the annual severity index
# =============================================================================

def compute_duration_decomposition(
    compound, daily_deficit, wcf_da, scf_da, ref_start, ref_end,
    duration_classes=DECOMPOSITION_DURATION_CLASSES, compute_severity_index=True,
):
    """
    Annual (year, lat, lon) drought-severity index (frequency * mean duration
    * severity), decomposed by event-duration class -- e.g. only single-day
    events, only 2-day events, events lasting 3+ days, and (as just another
    class) all events combined.

    `compound` is the plain 0/1 (time, lat, lon) "compound low-production
    day" field (rolling-mean-smoothed or raw daily -- the caller's pipeline
    decides), not land-masked. `daily_deficit` is the matching per-day
    severity field (sum of the wind/solar deficits below threshold) used to
    average over each duration class's qualifying days. `wcf_da`/`scf_da` are
    the raw (non-rolled) capacity-factor fields, used only to build the
    resource/land validity mask from reference-period non-NaN coverage.

    Returns ({label: annual_index_DataArray}, resource_valid, freq_all,
    freq_by_class), where freq_all is the "all events" class's annual
    event-count array -- exposed so callers can replicate fig1.py's
    build_land_mask convention (exclude pixels with zero events in the
    first on-record year) if needed -- and freq_by_class is
    {label: annual_frequency_DataArray} for every class (freq_all ==
    freq_by_class["all"]), letting callers decompose frequency alone
    (e.g. fig_red_decomposition.py's frequency value-by-alpha, which fixes
    duration per panel and so has no use for the duration/severity
    factors). frequency/duration here are 0-filled rather than NaN for
    no-event pixel-years, so that check has to be `== 0`, not `.isnull()`.

    `compute_severity_index=False` skips building the (expensive,
    full-grid-resample) severity index altogether -- for callers that only
    want freq_by_class -- and `indices` comes back empty in that case.
    """
    print("  Building event table of compound low-production spells")
    df_events = compute_event_table(compound)
    df_events_dedup = df_events.drop_duplicates(["event_id", "lat", "lon"])

    print("  Building resource/land validity mask (reference-period non-NaN wcf & scf)")
    wcf_ref_mean = wcf_da.sel(time=slice(ref_start, ref_end)).mean("time")
    scf_ref_mean = scf_da.sel(time=slice(ref_start, ref_end)).mean("time")
    resource_valid = (wcf_ref_mean.notnull() & scf_ref_mean.notnull()).astype("int8").load()

    indices = {}
    freq_by_class = {}
    freq_all = None
    for label, duration_class in duration_classes:
        print(f"  Computing decomposed annual stats for class '{label}'")
        freq, dur = compute_annual_stats_for_duration_class(
            df_events_dedup, compound, duration_class,
        )
        freq_by_class[label] = freq.load()
        if compute_severity_index:
            class_mask = build_duration_class_mask(df_events, compound, duration_class)
            severity = xr.where(class_mask, daily_deficit, np.nan).resample(time="YE").mean()
            severity["time"] = severity.time.dt.year
            # 0-fill (not NaN) for pixel-years with no qualifying event --
            # a "no drought" year is real, known data.
            severity = severity.rename({"time": "year"}).fillna(0.0)
            indices[label] = (freq_by_class[label] * dur * severity).load()
        if duration_class == ("ge", 1):
            freq_all = freq_by_class[label]

    return indices, resource_valid, freq_all, freq_by_class
