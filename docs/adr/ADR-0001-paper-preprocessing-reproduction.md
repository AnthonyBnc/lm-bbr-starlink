# ADR-0001: Paper preprocessing reproduction

## Status

Approved for an exploratory reproduction on 2026-08-06.

## Context

The SLM-BBR paper defines the phase detector, reward constants, feasible gains,
and surrogate models, but its released repositories do not include the claimed
preprocessing implementation or processed experience pool. The paper specifies
`w = 10` as the half-window for centered rolling percentile references but does
not separately state windows for the phase rolling mean, rolling RTT minimum, or
rolling retransmission maximum.

## Decision

With explicit user approval, the exploratory reproduction uses `w = 10` as the
half-window for all centered rolling calculations. A sample therefore uses up
to 21 observations: ten before, itself, and ten after.

The paper's state order is fixed as location flag, stream flag, time,
throughput, retransmissions, congestion window, receiver window, RTT, and RTT
variance. Development location flags follow the contract order: Ohio,
SaoPaulo, London, Mumbai, Sydney. Stream flags are downlink sequential, uplink
sequential, downlink competitive, and uplink competitive. Tokyo has no
development flag and is rejected by the generator.

Each raw capture remains a separate episode so rolling calculations and returns
cannot cross an iperf3 run boundary. Local extrema use adjacent samples, and
population standard deviation is used for the deviation threshold. With explicit
user approval on 2026-08-06, expert labels use the paper's continuous
utilization rule from Equations 2-3 followed by nearest phase-safe
discretization. Equation 16, backed by the throughput and retransmission
surrogates from Equations 22-27, is then used to assign the selected action's
reward. These implementation details are recorded because the paper and
released code do not resolve them more precisely.

## Consequences

Outputs use dataset version
`slm-bbr-paper-w10-eq2-3-labels-v1-exploratory`. They are not eligible for
headline results until their phase/action distributions are reviewed and the
reproduction assumptions are accepted for the final protocol. All models must
consume the exact same generated pool and sample IDs.

The earlier one-file Ohio smoke pool contained 301 samples and valid UP, DOWN,
and CRUISE phases, but literal Equation 16 optimization combined with
Equations 22-27 selected only three actions: 0.98, 1.00, and 1.05. That collapse
is why the approved exploratory reproduction now treats Equations 2-3 as the
authoritative expert-label construction and keeps Equation 16 as the reward
calculation for the selected label.

The full five-location exploratory pool was generated on 2026-08-06 from all
200 discovered non-Tokyo BBR traces. It contains 60,084 samples: 87.50% CRUISE,
5.48% DOWN, and 7.02% UP. All DOWN labels are 0.90 because Equation 3 produces
continuous targets between 0.50 and 0.820765 for the detected DOWN samples,
which are all nearest to the lowest permitted gain. This distribution is a
documented reproduction outcome, not permission to alter Equation 3 or rebalance
headline data. Training remains paused pending review of this imbalance and a
frozen development-only validation split.
