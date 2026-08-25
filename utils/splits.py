"""Deterministic trace-level splits for Starlink development data."""

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import PurePosixPath

from utils.exp_pool import ExperiencePool
from utils.starlink_preprocessing import (
    DEVELOPMENT_LOCATIONS,
    HELD_OUT_LOCATION,
    STREAM_FLAGS,
)


def _trace_slices(pool):
    source_files = pool.metadata.get("source_files")
    if not source_files:
        raise ValueError("Experience pool metadata has no source_files")

    slices = []
    start = 0
    for source in source_files:
        intervals = int(source.get("intervals", 0))
        if intervals < 1:
            raise ValueError("Invalid interval count for {}".format(source.get("path")))
        end = start + intervals
        if end > len(pool):
            raise ValueError("Source interval counts exceed experience-pool length")
        if not pool.dones[end - 1] or any(pool.dones[start : end - 1]):
            raise ValueError("Done flags do not match source trace {}".format(source["path"]))
        slices.append((source, start, end))
        start = end
    if start != len(pool):
        raise ValueError(
            "Source interval counts cover {} samples, but pool has {}".format(start, len(pool))
        )
    return slices


def _trace_identity(source):
    parts = PurePosixPath(source["path"]).parts
    if len(parts) < 3:
        raise ValueError("Unexpected source trace path: {}".format(source["path"]))
    stream, location = parts[0], parts[1]
    if location == HELD_OUT_LOCATION:
        raise ValueError("Tokyo cannot enter a development split")
    if location not in DEVELOPMENT_LOCATIONS:
        raise ValueError("Unknown development location: {}".format(location))
    if stream not in STREAM_FLAGS:
        raise ValueError("Unknown stream group: {}".format(stream))
    return location, stream


def _subset_pool(pool, selected_paths, role, split_version):
    subset = ExperiencePool()
    selected_sources = []
    for source, start, end in _trace_slices(pool):
        if source["path"] not in selected_paths:
            continue
        selected_sources.append(deepcopy(source))
        for index in range(start, end):
            subset.add(
                state=pool.states[index],
                action=pool.actions[index],
                reward=pool.rewards[index],
                done=pool.dones[index],
                phase=pool.phases[index],
                sample_id=pool.sample_ids[index],
            )

    subset.metadata = deepcopy(pool.metadata)
    subset.metadata.update(
        {
            "parent_dataset_version": pool.metadata.get("dataset_version"),
            "dataset_version": "{}-{}".format(split_version, role),
            "split_role": role,
            "split_version": split_version,
            "source_files": selected_sources,
        }
    )
    return subset


def stratified_trace_split(pool, validation_traces_per_stratum=2, seed=100003):
    """Split traces within every development location and stream stratum."""
    if validation_traces_per_stratum < 1:
        raise ValueError("validation_traces_per_stratum must be at least 1")

    strata = defaultdict(list)
    for source, _, _ in _trace_slices(pool):
        identity = _trace_identity(source)
        strata[identity].append(source["path"])

    expected_strata = {
        (location, stream)
        for location in DEVELOPMENT_LOCATIONS
        for stream in STREAM_FLAGS
    }
    if set(strata) != expected_strata:
        missing = sorted(expected_strata - set(strata))
        extra = sorted(set(strata) - expected_strata)
        raise ValueError("Development strata mismatch; missing={}, extra={}".format(missing, extra))

    validation_paths = set()
    stratum_records = []
    for (location, stream), paths in sorted(strata.items()):
        if validation_traces_per_stratum >= len(paths):
            raise ValueError("Validation would consume every trace in {} / {}".format(location, stream))
        ranked = sorted(
            paths,
            key=lambda path: (sha256("{}:{}".format(seed, path).encode("utf-8")).hexdigest(), path),
        )
        chosen = sorted(ranked[:validation_traces_per_stratum])
        validation_paths.update(chosen)
        stratum_records.append(
            {
                "location": location,
                "stream": stream,
                "train_traces": len(paths) - validation_traces_per_stratum,
                "validation_traces": validation_traces_per_stratum,
                "validation_paths": chosen,
            }
        )

    all_paths = {source["path"] for source, _, _ in _trace_slices(pool)}
    train_paths = all_paths - validation_paths
    split_version = "dev-trace-stratified-80-20-seed-{}-v1-exploratory".format(seed)
    train_pool = _subset_pool(pool, train_paths, "train", split_version)
    validation_pool = _subset_pool(pool, validation_paths, "validation", split_version)
    manifest = {
        "split_version": split_version,
        "status": "exploratory_pending_supervisor_approval",
        "seed": seed,
        "method": "SHA256-ranked trace holdout within each location x stream stratum",
        "validation_traces_per_stratum": validation_traces_per_stratum,
        "held_out_location": HELD_OUT_LOCATION,
        "train_traces": len(train_paths),
        "validation_traces": len(validation_paths),
        "train_samples": len(train_pool),
        "validation_samples": len(validation_pool),
        "strata": stratum_records,
    }
    return train_pool, validation_pool, manifest
