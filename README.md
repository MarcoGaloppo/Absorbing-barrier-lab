# Statistics in human affairs — a tennis laboratory

[![CI](https://github.com/MarcoGaloppo/Absorbing-barrier-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoGaloppo/Absorbing-barrier-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## In a nutshell

We build tennis models to investigate serial correlation, long memory, and absorbing barriers. Short-term correlation directly inflates volatility and long memory is surprisingly easy to detect.
Sadly, estimating its effect is near impossible under real-wolrd circumstances. Finally, in the presence of absorbing barriers these are what sets the extremes.

## To wit

How much of textbook statistics' comfortable machinery survives contact with a
process that has **absorbing barriers**, **serial correlation**, and **long memory**?
Tennis is a clean (and fun!) place to ask, because the bottom of the process really is a
Bernoulli trial. Additionally, everything above it is bookkeeping that we control exactly.
Although finance is messy, it is sometimes good to start from a ground truth and build up
some basic intuition — without being too fooled by it.

## The set-up

Player A wins any point with probability `p`. Under textbook assumptions the number of points
A wins in `n` points is exactly `Binomial(n, p)` and `p̂` has standard error `√(p(1-p)/n)`.
Here, we break this simple scenario in various ways:

**1. Nonlinear aggregation.** Tennis counts games (first to 4, win by 2), sets (first to 6
games, win by 2, tiebreak at 6–6), and matches (best of 3 or 5). Every layer is an effective
absorbing-barrier stopping problem, and the composition `f(p) = P(A wins the match)` is a
steep S-curve. It is trivial to say, but scoring a point and winning a game are not the
same thing. And they do not carry the same probability given a set chance of winning a point.
**Note, this is present even with perfectly i.i.d. points.**

**2. Serial correlation.** The point-win probability depends on the previous point via

(1) p_t = σ( logit(p) + c + k · s_{t-1} ),      s_{t-1} ∈ {−1, +1}.

The offset `c` is solved numerically so the **stationary** win probability stays exactly `p`.
This is done because turning correlation on then changes the variance and path structure
**without moving the mean**. Also, note that to try and avoid exiting completely from
CLT territory we consider short-time correlations within each game (at this stage).

**3. Long memory, specified through the run-length hazard.**  What actually
controls the memory is the tail of the run-length distribution, so we specify that directly to
try to introduce fat tails and long memory:

(2) h(ℓ) = P(run ends next | run has reached ℓ) = β / (ℓ + γ).

The survival function behaves as `P(L > ℓ) ~ C ℓ^(−β)`, so β is literally a Pareto tail index.
For `1 < β < 2` the run variance is infinite, `Σ_h ρ_h` diverges, `Var(S_n) ~ n^(3−β)`, and the failure is
in the scaling *exponent* of the variance. That is the theory. When fat tails meet absorbing barriers,
however, things can change drastically.

## Headline results

1. **A hot hand makes us overconfident.**
An order-1 chain (short-term memory), effectively generates excess kurtosis which still decays
to zero. Correlation of this kind does not buy fat tails; it buys only **overconfidence**. And
this is a problem. Because, then, every normality test on correctly-standardised residuals will pass
while your error bars stay too small with respect to the actual scenario you would like to describe.
There is more volatility in the system, even in this case. Quite fun!

2. **Momentum is a variance pump, and in a barrier-crossing game variance is worth something to whoever is behind.**
We have an asymmetric situation in which the (relatively) close-to-win-but-still-losing player benefits more from luck (i.e., more things can go better than worse for them) than the close-to-losing-but-still-winning player, who has a more capped ceiling on good luck than bad luck. At p = 0.55 held fixed, the favourite's match-win probability falls from 91% to 77% as k goes from 0 to 1. **A model fitted on point-level data and aggregated under independence systematically overprices favourites.** If a player is a (slight) underdog, volatility will benefit them more!

3. **The absorbing barriers kill the long memory in tennis.**
We consider the underlying chosen models within the constraint of *real matches*. As such, a match starts at the first point
and ends when someone has won two sets. The longest run physically possible in a best-of-3 is **53** points.
We found that by keeping the mean run length fixed at 4.0 and sweeping β from 3.5 down to 1.3 — i.e., from finite
to infinite variance — the longest run ever observed is **52 or 53 every single time**. In other words: the
extreme is pinned to the format, not to the underlying process. In fact, the clipped runs would be those that carry
*all* of the variance for β < 2.0. As such we discovered the simple fact that **a truncated power law is not a power law**.

4. **Estimating point and match win probabilities is not the same thing.**
The mapping from the probability of winning a point to winning a match is strikingly nonlinear. A player that wins 51% of the points wins about 60.6% of the best-of-3 matches. With 55% points won, this translates to over 90% of the best-of-3 matches. The absorbing barriers are clearly carrying this nonlinear mapping. The nonlinearity
carries over fragility to the estimation of the win-point probability. As a consequence, even if the probability of winning a single point can be estimated
reasonably well with a single game, this translates into completely uninformative posteriors on winning the match.

5. **Detecting long memory is easy; measuring its consequences is impossible.** Given a single match, can we detect long memory? And if so,
can we say anything about the Pareto exponent? Well, it turns out that detection is startlingly easy, with essentially 100% detection rates from two matches and 80%
with a single one, at a 0–1% false-positive rate. This is because the discrimination power between short and long memory is carried by the **body**
of the run-length distribution (e.g., runs of 5–20 points, which fit comfortably inside a match) rather than by the tail the barrier destroyed. However, **the barrier annihilates the predictive power on the Pareto exponent**. Indeed, the 95% interval on β from one match is `[1.1, 5.9]`, and it takes
about twenty-five matches merely to place β on one side of the phase boundary at 2.

## Files

1. `tennis_lab.py`: The simulator. `TennisSimulator` (point → game → set → match, with deuce and tiebreak, optional order-1 momentum) plus exact closed forms `game_win_probability`, `set_win_probability`, `match_win_probability_iid`. Depends only on `numpy`.
2. `narrative_tennis_lab.ipynb`: The analysis: eight sections, with narrative. Outputs are committed, so it reads on GitHub without running.
3. `tests/test_tennis_lab.py`: every claim in *Quick Validations* below, executed under CI.

Key knobs to turn: `p` (target stationary point-win probability), `k` (momentum strength in
log-odds; 0 = i.i.d., 1.0 = strong hot hand), `best_of`, `calibrate`, `seed`. For the
long-memory regime: `memory="long"`, `beta` (tail index) and `gamma` (scale, most usefully set
via `gamma_for_mean_run`).

Quantities that are theorems about the order-1 chain (`rho`, `variance_inflation`,
`exact_variance`, `effective_sample_size_ratio`) raise `NotImplementedError` under
`memory="long"` rather than returning a plausible-looking wrong number, and point at
`hurst_exponent` / `run_tail_index` instead.

## Quick Validations

The module is checked against closed forms:

- `k = 0` reproduces `Binomial(n, p)` to Monte-Carlo error.
- Simulated variance matches `exact_variance(n)` — the untruncated covariance sum
  `p(1−p)[n + 2Σ(n−h)ρ^h]` — across `k ∈ [0, 1.6]`.
- Simulated match-win frequency matches `match_win_probability_iid` (0.9089 vs 0.9100 on
  30,000 best-of-3 matches; 0.9526 vs 0.9530 on best-of-5).
- `stationary_p` is pinned at the target `p` to six decimals for every `k`.
- Empirical run-length survival matches the exact Gamma-ratio formula `run_survival`
  (0.03565 vs 0.03556 for `P(L > 10)` at β = 2.5).
- `gamma_for_mean_run` reproduces `E[L_A]` to machine precision, and the closed-form
  calibration `p = γ_A/(γ_A+γ_B)` holds for every β.
- `simulate_match_points` returns exactly `total_points` outcomes per match, so recording
  provably does not perturb the random stream.
- The 53-point ceiling was confirmed by exhaustive search over 150,000 deliberately streaky
  matches (77 for best-of-5), never once exceeded.

Each of these is a test in `tests/`, so the list above is enforced rather than asserted.

## Running it

`numpy` for the module; `pandas`, `matplotlib`, `jupyter` additionally for the notebook.
No `scipy` — the binomial pmf and the normal quantile function are implemented inline.

```
pip install -r requirements.txt
jupyter lab narrative_tennis_lab.ipynb

```
## On the use of AI

`tennis_lab.py` and the notebook was written with Claude (Anthropic) used as a pair programmer
over multiple sessions. I chose the questions, the structure, and the standard of evidence whilst 
employing Claude to write a good part of the implementation and some of the prose.

Every numerical claim in the notebooks is produced by the code in this repository and was checked
against its output. Several of Claude's results were wrong and were corrected appropriately, for 
example ignoring the absorbing barrier in the long-memory case.

## Tests

```
pip install pytest
pytest                 # ~3 s
pytest --runslow       # adds the exhaustive barrier-ceiling searches, ~15 s
```

The suite is seeded throughout and its Monte-Carlo tolerances sit at several standard
errors, so a red run means a regression rather than an unlucky seed. Two of the tests are
a matched pair worth pointing at: `test_runs_are_unbounded_without_a_scoreboard` and
`test_longest_run_is_capped_by_the_format` run the *same* β = 1.3 process and get runs of
several hundred points in one case and never more than 53 in the other. That is headline
result 3, as an assertion.

## License

MIT — see [LICENSE](LICENSE).

Author: Marco Galoppo
