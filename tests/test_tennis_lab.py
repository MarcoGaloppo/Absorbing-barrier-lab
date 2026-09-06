"""
The claims the README makes, executed.

Every validation listed under "Quick Validations" in the README lives here as a
test. Monte-Carlo tests are seeded and their tolerances are pre-set.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tennis_lab import (
    TennisSimulator,
    calibrated_transitions,
    check_tail_params,
    game_win_probability,
    gamma_for_mean_run,
    hurst_from_beta,
    logit,
    loglik_markov,
    markov_stats,
    match_win_probability_iid,
    race_win_probability,
    run_hazard,
    run_stats,
    run_survival,
    set_win_probability,
    sigmoid,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def completed_runs(paths: np.ndarray, who: int = 1) -> np.ndarray:
    """
    Lengths of the fully observed runs of ``who`` in each path.

    The first run is dropped because the chain starts from a state that already
    implies one point of history, and the last is dropped because the sequence
    stops before the run does. Both would bias the length distribution downward.
    """
    out = []
    for x in np.atleast_2d(paths):
        x = np.asarray(x, dtype=np.int8)
        change = np.flatnonzero(np.diff(x)) + 1
        starts = np.concatenate(([0], change))
        lens = np.concatenate((change, [x.size])) - starts
        owner = x[starts]
        if len(lens) < 3:
            continue
        keep = (owner[1:-1] == who)
        out.append(lens[1:-1][keep])
    return np.concatenate(out) if out else np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# 1. k = 0 reproduces Binomial(n, p)
# --------------------------------------------------------------------------- #
def test_iid_points_are_binomial():
    n, p = 100, 0.55
    sim = TennisSimulator(p=p, memory="none", seed=0)
    counts = sim.simulate_point_sequences(20_000, n)

    assert counts.mean() == pytest.approx(n * p, abs=0.2)
    assert counts.std() == pytest.approx(math.sqrt(n * p * (1 - p)), abs=0.15)
    # a Binomial has no excess kurtosis to speak of at this n
    z = (counts - counts.mean()) / counts.std()
    assert abs((z ** 4).mean() - 3.0) < 0.3


# --------------------------------------------------------------------------- #
# 2. simulated variance matches exact_variance(n) across k
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [0.0, 0.4, 0.8, 1.2, 1.6])
def test_variance_matches_closed_form(k):
    n = 200
    sim = TennisSimulator(p=0.55, k=k, memory="markov", seed=11)
    counts = sim.simulate_point_sequences(40_000, n)
    assert counts.var() == pytest.approx(sim.exact_variance(n), rel=0.05)


def test_variance_inflation_is_the_asymptote_from_above():
    """exact_variance/(n p q) rises toward (1+rho)/(1-rho) but never passes it."""
    sim = TennisSimulator(p=0.55, k=1.0, memory="markov", seed=0)
    pq = sim.p * (1 - sim.p)
    ratios = [sim.exact_variance(n) / (n * pq) for n in (50, 200, 1000, 5000)]
    assert ratios == sorted(ratios)
    assert ratios[-1] < sim.variance_inflation
    assert ratios[-1] == pytest.approx(sim.variance_inflation, rel=0.02)


# --------------------------------------------------------------------------- #
# 3. simulated match-win frequency matches the closed form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("best_of", [3, 5])
def test_match_win_frequency_matches_closed_form(best_of):
    p = 0.55
    sim = TennisSimulator(p=p, memory="none", best_of=best_of, seed=2)
    freq = sim.match_win_probability(6_000)
    assert freq == pytest.approx(match_win_probability_iid(p, best_of), abs=0.02)


# --------------------------------------------------------------------------- #
# 4. stationary_p is pinned at the target for every k
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [0.0, 0.25, 0.5, 1.0, 1.6, -0.7])
@pytest.mark.parametrize("p", [0.45, 0.5, 0.55, 0.7])
def test_calibration_pins_the_stationary_mean(p, k):
    sim = TennisSimulator(p=p, k=k, memory="markov", calibrate=True)
    assert sim.stationary_p == pytest.approx(p, abs=1e-6)


def test_uncalibrated_momentum_moves_the_mean():
    """The reason calibrate exists: without it, k shifts the mean as well."""
    sim = TennisSimulator(p=0.55, k=1.0, memory="markov", calibrate=False)
    assert abs(sim.stationary_p - 0.55) > 1e-3


def test_calibrated_transitions_agrees_with_the_simulator():
    for k in (0.3, 0.9, 1.5):
        sim = TennisSimulator(p=0.55, k=k, memory="markov")
        a, b = calibrated_transitions(0.55, k)
        assert (a, b) == pytest.approx(sim.transition_probs, rel=1e-9)


# --------------------------------------------------------------------------- #
# 5. run-length survival: closed form, and the sampler that must obey it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta,gamma", [(1.3, 1.2), (1.8, 3.0), (2.5, 6.0), (3.5, 9.0)])
def test_run_survival_equals_the_hazard_product(beta, gamma):
    ell = np.arange(1, 40)
    product = np.cumprod(1.0 - run_hazard(np.arange(1, 40), beta, gamma))
    assert run_survival(ell, beta, gamma) == pytest.approx(product, rel=1e-12)


def test_run_survival_has_the_advertised_pareto_tail():
    """P(L > l) ~ C l^(-beta): the local slope on log-log approaches -beta."""
    beta, gamma = 2.5, 6.0
    ell = np.array([2_000.0, 4_000.0])
    s = run_survival(ell, beta, gamma)
    slope = np.diff(np.log(s)) / np.diff(np.log(ell))
    assert slope[0] == pytest.approx(-beta, abs=0.02)


def test_sampler_run_lengths_match_the_closed_form():
    beta, mean_run, p = 2.5, 4.0, 0.5
    g = gamma_for_mean_run(mean_run, beta, p)
    sim = TennisSimulator(p=p, memory="long", beta=beta, gamma=g, seed=3)
    _, paths = sim.simulate_point_sequences(3_000, 400, return_paths=True)

    lengths = completed_runs(paths, who=1)
    assert lengths.size > 50_000

    gamma_a = sim.hazard_params[1]
    for ell in (1, 2, 5, 10, 20):
        empirical = float((lengths > ell).mean())
        assert empirical == pytest.approx(
            float(run_survival(ell, beta, gamma_a)), abs=0.005
        )


# --------------------------------------------------------------------------- #
# 6. gamma_for_mean_run, and the closed-form p = gamma_A / (gamma_A + gamma_B)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", [1.2, 1.5, 1.9, 2.5, 3.5])
@pytest.mark.parametrize("p", [0.45, 0.5, 0.55, 0.6])
def test_gamma_for_mean_run_round_trips(p, beta):
    target = 4.0
    g = gamma_for_mean_run(target, beta, p)
    sim = TennisSimulator(p=p, memory="long", beta=beta, gamma=g)
    mean_a, _ = sim.mean_run_lengths
    assert mean_a == pytest.approx(target, rel=1e-12)

    _, gamma_a, gamma_b = sim.hazard_params
    assert gamma_a / (gamma_a + gamma_b) == pytest.approx(p, rel=1e-12)


def test_mean_run_length_converges_as_the_window_grows():
    """
    A finite window is itself a mild barrier: the runs it clips are the long
    ones, so the mean of the *completed* runs comes in low and only converges
    to gamma_A/(beta-1) as the window opens up. Same mechanism as headline
    result 3, without the scoreboard.
    """
    beta, target, p = 2.5, 4.0, 0.5
    g = gamma_for_mean_run(target, beta, p)

    def mean_run(n_points, n_seq):
        sim = TennisSimulator(p=p, memory="long", beta=beta, gamma=g, seed=4)
        _, paths = sim.simulate_point_sequences(n_seq, n_points, return_paths=True)
        return completed_runs(paths, who=1).mean()

    short, long = mean_run(400, 2_000), mean_run(4_000, 600)
    assert short < long < target
    assert long == pytest.approx(target, rel=0.015)


def test_runs_are_unbounded_without_a_scoreboard():
    """
    The counterpart to test_longest_run_is_capped_by_the_format: the very same
    process, left to run freely, produces runs an order of magnitude past the
    53 the match format allows. The cap is the format, not the process.
    """
    g = gamma_for_mean_run(4.0, beta=1.3, p=0.5)
    sim = TennisSimulator(p=0.5, memory="long", beta=1.3, gamma=g, seed=12)
    _, paths = sim.simulate_point_sequences(200, 20_000, return_paths=True)
    assert completed_runs(paths, who=1).max() > 200


# --------------------------------------------------------------------------- #
# 7. recording does not perturb the random stream
# --------------------------------------------------------------------------- #
def test_recording_does_not_perturb_the_stream():
    kwargs = dict(p=0.55, k=0.8, memory="markov", seed=7)
    totals = TennisSimulator(**kwargs).simulate_matches(50)["total_points"]
    recorded = TennisSimulator(**kwargs).simulate_match_points(50)
    assert [len(a) for a in recorded] == list(totals)


def test_recording_is_cleared_afterwards():
    sim = TennisSimulator(p=0.55, seed=1)
    sim.simulate_match_points(3)
    assert sim._record is None


# --------------------------------------------------------------------------- #
# 8. the absorbing barrier caps the extreme, whatever beta says
# --------------------------------------------------------------------------- #
def test_longest_run_is_capped_by_the_format():
    """
    53 points is the longest run physically possible in a best-of-3. The tail
    index of the underlying process cannot change that, which is the whole
    point of headline result 3.
    """
    g = gamma_for_mean_run(4.0, beta=1.3, p=0.5)
    sim = TennisSimulator(p=0.5, memory="long", beta=1.3, gamma=g, best_of=3, seed=5)
    runs = sim.simulate_matches(5_000)["longest_run_a"]
    assert runs.max() <= 53


@pytest.mark.slow
def test_longest_run_ceiling_exhaustively():
    for best_of, ceiling in ((3, 53), (5, 77)):
        g = gamma_for_mean_run(4.0, beta=1.3, p=0.5)
        sim = TennisSimulator(
            p=0.5, memory="long", beta=1.3, gamma=g, best_of=best_of, seed=6
        )
        runs = sim.simulate_matches(50_000)["longest_run_a"]
        assert runs.max() <= ceiling


@pytest.mark.slow
def test_ceiling_is_insensitive_to_beta():
    """Sweeping beta across the infinite-variance boundary moves nothing."""
    maxima = []
    for beta in (3.5, 2.5, 1.8, 1.3):
        g = gamma_for_mean_run(4.0, beta=beta, p=0.5)
        sim = TennisSimulator(p=0.5, memory="long", beta=beta, gamma=g, seed=8)
        maxima.append(int(sim.simulate_matches(20_000)["longest_run_a"].max()))
    assert all(m <= 53 for m in maxima)
    assert max(maxima) - min(maxima) <= 3


# --------------------------------------------------------------------------- #
# closed forms: symmetry, monotonicity, amplification
# --------------------------------------------------------------------------- #
def test_fair_points_give_fair_everything():
    assert game_win_probability(0.5) == pytest.approx(0.5, abs=1e-12)
    assert set_win_probability(0.5) == pytest.approx(0.5, abs=1e-12)
    for best_of in (3, 5):
        assert match_win_probability_iid(0.5, best_of) == pytest.approx(0.5, abs=1e-12)


def test_aggregation_is_monotone_and_amplifying():
    ps = np.linspace(0.40, 0.60, 21)
    f = np.array([match_win_probability_iid(p, 3) for p in ps])
    assert np.all(np.diff(f) > 0)
    # a 1-point edge becomes a 10-point edge; a 5-point edge becomes 40
    assert match_win_probability_iid(0.51, 3) > 0.58
    assert match_win_probability_iid(0.55, 3) > 0.88
    # best-of-5 amplifies harder than best-of-3
    assert match_win_probability_iid(0.55, 5) > match_win_probability_iid(0.55, 3)


def test_race_win_probability_reduces_to_a_game():
    assert race_win_probability(0.6, target=4) == pytest.approx(
        game_win_probability(0.6, 4), rel=1e-12
    )
    with pytest.raises(NotImplementedError):
        race_win_probability(0.6, target=4, margin=3)


# --------------------------------------------------------------------------- #
# hurst / tail bookkeeping
# --------------------------------------------------------------------------- #
def test_hurst_from_beta():
    assert hurst_from_beta(3.5) == 0.5
    assert hurst_from_beta(2.0) == 0.5
    assert hurst_from_beta(1.5) == pytest.approx(0.75)
    assert 0.5 < hurst_from_beta(1.9) < 1.0
    with pytest.raises(ValueError):
        hurst_from_beta(1.0)


def test_check_tail_params_guards_the_gamma_beta_relation():
    with pytest.raises(ValueError):
        check_tail_params(beta=-1.0, gamma=2.0)
    with pytest.raises(ValueError):
        check_tail_params(beta=2.5, gamma=1.0)      # gamma <= beta - 1
    check_tail_params(beta=2.5, gamma=1.6)          # fine


def test_long_memory_rejects_beta_below_one():
    with pytest.raises(ValueError):
        TennisSimulator(memory="long", beta=0.9, gamma=5.0)


# --------------------------------------------------------------------------- #
# the regime guards: wrong answers are refused, not returned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attr", ["rho", "variance_inflation", "effective_sample_size_ratio",
             "transition_probs", "mean_run_lengths", "hazard_params"]
)
def test_markov_only_quantities_raise_under_long_memory(attr):
    g = gamma_for_mean_run(4.0, beta=1.6, p=0.55)
    sim = TennisSimulator(p=0.55, memory="long", beta=1.6, gamma=g)
    if attr in ("mean_run_lengths", "hazard_params"):
        getattr(sim, attr)                       # these are the long-memory ones
    else:
        with pytest.raises(NotImplementedError):
            getattr(sim, attr)


def test_long_only_quantities_raise_under_the_order_one_chain():
    sim = TennisSimulator(p=0.55, k=0.5, memory="markov")
    for attr in ("run_tail_index", "mean_run_lengths", "hazard_params"):
        with pytest.raises(NotImplementedError):
            getattr(sim, attr)
    assert sim.hurst_exponent == 0.5


def test_constructor_validation():
    with pytest.raises(ValueError):
        TennisSimulator(p=0.0)
    with pytest.raises(ValueError):
        TennisSimulator(memory="markovian")
    with pytest.raises(ValueError):
        TennisSimulator(best_of=4)


# --------------------------------------------------------------------------- #
# regressions
# --------------------------------------------------------------------------- #
def test_repr_works_in_both_regimes():
    """repr used to call rho, which raises under long memory."""
    assert "k=" in repr(TennisSimulator(p=0.55, k=0.5, memory="markov"))

    g = gamma_for_mean_run(4.0, beta=1.6, p=0.55)
    text = repr(TennisSimulator(p=0.55, memory="long", beta=1.6, gamma=g))
    assert "long" in text and "beta=1.600" in text


def test_summary_works_in_both_regimes():
    assert "rho" in TennisSimulator(p=0.55, k=0.5, memory="markov").summary()
    g = gamma_for_mean_run(4.0, beta=1.6, p=0.55)
    keys = TennisSimulator(p=0.55, memory="long", beta=1.6, gamma=g).summary()
    assert keys["hurst"] == pytest.approx(0.7)


def test_with_params_carries_beta_gamma_and_seed():
    """with_params used to drop beta, gamma and seed on the floor."""
    g = gamma_for_mean_run(4.0, beta=1.6, p=0.55)
    sim = TennisSimulator(p=0.55, memory="long", beta=1.6, gamma=g, seed=42)
    child = sim.with_params(p=0.60)

    assert child.beta == 1.6
    assert child.gamma == g
    assert child.seed == 42
    assert child.p == 0.60
    assert child.memory == "long"

    # same seed means the same stream, not an arbitrary new one
    twin = TennisSimulator(p=0.60, memory="long", beta=1.6, gamma=g, seed=42)
    assert np.array_equal(
        child.simulate_point_sequences(20, 50), twin.simulate_point_sequences(20, 50)
    )


def test_with_params_round_trips_the_markov_regime():
    sim = TennisSimulator(p=0.55, k=0.9, memory="markov", best_of=5, seed=1)
    child = sim.with_params(k=0.3)
    assert child.best_of == 5 and child.seed == 1 and child.k == 0.3
    assert child.stationary_p == pytest.approx(0.55, abs=1e-6)


def test_sigmoid_does_not_truncate_integer_input():
    """np.empty_like on an int array used to round 0.5 down to 0."""
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(np.array([0, 1])) == pytest.approx([0.5, 1 / (1 + math.exp(-1))])


def test_logit_sigmoid_round_trip():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert sigmoid(logit(p)) == pytest.approx(p, rel=1e-12)


# --------------------------------------------------------------------------- #
# inference plumbing
# --------------------------------------------------------------------------- #
def test_markov_loglik_is_maximised_near_the_truth():
    k_true = 0.8
    sim = TennisSimulator(p=0.55, k=k_true, memory="markov", seed=9)
    _, paths = sim.simulate_point_sequences(400, 500, return_paths=True)
    stats = markov_stats(paths)

    grid = np.linspace(0.0, 1.6, 33)
    ll = [loglik_markov(stats, 0.55, k) for k in grid]
    assert grid[int(np.argmax(ll))] == pytest.approx(k_true, abs=0.1)


def test_run_stats_censors_the_final_run():
    sim = TennisSimulator(p=0.55, k=0.5, memory="markov", seed=10)
    _, paths = sim.simulate_point_sequences(200, 300, return_paths=True)
    a_done, b_done, a_cens, b_cens = run_stats(list(paths))
    # exactly one censored run per path, and it belongs to whoever won last
    assert a_cens.sum() + b_cens.sum() == 200
    assert a_done.sum() + b_done.sum() > 0
