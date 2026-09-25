# ADR-0002: Exploratory development validation split

## Status

Approved by the user for exploratory implementation on 2026-08-06. Pending
supervisor approval before headline experiments.

## Context

Tokyo is the held-out location and cannot be used for checkpoint selection or
hyperparameter decisions. The five development locations each contain ten BBR
traces for every downlink/uplink and sequential/competitive stream group. A
sample-level random split would leak adjacent observations from the same trace
across training and validation.

## Decision

Use a deterministic trace-level 80/20 split inside every location by stream
stratum. Rank trace paths by SHA-256 of seed `100003` and the path, then assign
the first two of ten traces to validation and the remaining eight to training.
This produces 160 training traces and 40 validation traces while retaining all
five development locations and four stream groups in both subsets.

The split version is
`dev-trace-stratified-80-20-seed-100003-v1-exploratory`. Tokyo is rejected, and
training and validation sample IDs must be disjoint. Model sequence windows
must remain inside one trace and cannot cross an episode-end flag.

## Consequences

The generated exploratory split contains 48,065 training samples and 12,019
validation samples. The full parent pool is preserved exactly as the disjoint
union of both subsets. All headline models must receive this same split if the
supervisor approves it. Until then, results using this split remain exploratory
and cannot enter the final comparison table.
