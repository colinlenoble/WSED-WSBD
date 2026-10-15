# -*- coding: cp1252 -*-
"""
Event table and event-duration-class decomposition of a 0/1 (time, lat, lon)
compound-day field: annual frequency, duration and severity restricted to
events of a given length (1 day, 2 days, 3+ days, ...).
"""
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Gap-tolerant event detection (event table)
# =============================================================================

def compute_event_table(da, time_dim="time"):
    """One row per (event day, lat, lon) for every run of True in `da`, with the
    event's start year and total 'duration'."""
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
    """(time, lat, lon) mask of days in events matching `duration_class`, an
    (op, value) pair with op 'eq' or 'ge' (e.g. ('ge', 3): 3+ days)."""
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
    """Annual event count and mean duration of events matching `duration_class`,
    0 where there are none."""
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


# Duration classes, label -> (op, value); "all" is ("ge", 1).
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
    """Annual severity index (frequency x duration x mean daily deficit) for each
    event-duration class.

    compound      : 0/1 (time, lat, lon) compound-day field, not land-masked
    daily_deficit : per-day wind + solar deficit below threshold
    wcf_da/scf_da : capacity factors, for the validity mask (non-NaN over the
                    reference period)

    Returns ({label: index}, resource_valid, freq_all, freq_by_class);
    frequency and duration are 0 (not NaN) without events.
    compute_severity_index=False skips the expensive index (indices empty)."""
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
            # Years without a qualifying event: 0, not NaN.
            severity = severity.rename({"time": "year"}).fillna(0.0)
            indices[label] = (freq_by_class[label] * dur * severity).load()
        if duration_class == ("ge", 1):
            freq_all = freq_by_class[label]

    return indices, resource_valid, freq_all, freq_by_class
