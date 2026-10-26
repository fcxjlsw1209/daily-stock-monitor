"""Causal price-correlated peers from the historical index-union research pool.

This is not industry breadth or a fully certified point-in-time universe. The
source pool retains retired names but inherits its documented available-data,
identity-continuity and provisional membership assumptions. It does not use the
current 503-stock universe or future labels. Missing evidence gives exact zero
correction and a false validity flag, without changing trading eligibility.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "index_union"
CORRELATION_RETURNS = 252
MA_SESSIONS = 200
BROADENING_LAG = 63
PRICE_OBSERVATIONS = MA_SESSIONS + BROADENING_LAG
MAX_PEERS = 10
MIN_PEERS = 5
FEATURE_NAMES = ("peer_participation", "peer_broadening", "market_participation", "market_broadening")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_peer_panel(root=ROOT):
    """Load and checksum existing research panels; never refresh remote data."""
    root = Path(root)
    manifest_path = root / "alias_resolved_inputs/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {str(manifest_path): _sha(manifest_path)}
    fields = {r["field"]: r for r in manifest["fields"]}
    data = {}
    for key in ("adj", "close", "eligible"):
        row = fields[key]
        path = (root / row["file"]).resolve()
        if root.resolve() not in path.parents or _sha(path) != row["sha256"]:
            raise ValueError("Peer source path/checksum mismatch: " + key)
        data[key] = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        hashes[str(path)] = row["sha256"]
    alias_path = root / "DATED_SYMBOL_IDENTITIES.json"
    aliases = json.loads(alias_path.read_text())["rows"]
    hashes[str(alias_path)] = _sha(alias_path)
    for row in aliases:
        if not row.get("retrospective_identity_link_verified"):
            raise ValueError("Unreviewed historical alias")
        evidence = (root / row["evidence_file"]).resolve()
        if root.resolve() not in evidence.parents or _sha(evidence) != row["evidence_sha256"]:
            raise ValueError("Alias evidence checksum mismatch")
        hashes[str(evidence)] = row["evidence_sha256"]
    for name in ("AVAILABLE_RESEARCH_ASSUMPTIONS.json", "BACKTEST_READINESS.json"):
        path = root / name
        hashes[str(path)] = _sha(path)
    data.update(issuers=dict(manifest["issuers"]), aliases=aliases,
                source_hashes=hashes,
                source_scope="historical_index_union_available_data_research",
                strict_complete_history_certified=False)
    return data


def _validate_frame(frame, label):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(label + " must be a DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(label + " must have DatetimeIndex exchange sessions")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.columns.is_unique:
        raise ValueError(label + " must have unique, ordered sessions and unique securities")


def _valid_issuer(value):
    return isinstance(value, str) and bool(value) and not value.startswith("unresolved_")


def _target_issuer(symbol, day, peers):
    """Resolve only explicit canonical IDs or verified date-bounded aliases.

    A research_equity ID is the already consolidated continuous security. Old
    and new ticker labels are usable only in their recorded historical epochs;
    do not guess punctuation conversions or splice reused symbols.
    """
    direct = peers["issuers"].get(symbol)
    if _valid_issuer(direct):
        return direct, "manifest_issuer"
    label = day.strftime("%Y-%m-%d")
    records = [r for r in peers.get("aliases", []) if r["symbol"] == symbol
               and r.get("retrospective_identity_link_verified")
               and r["effective_from"] <= label < r["effective_to"]]
    resolved = set()
    for record in records:
        issuer = peers["issuers"].get(record["security_id"])
        if _valid_issuer(issuer) and issuer == record["issuer_id"]:
            resolved.add(issuer)
    if len(resolved) == 1:
        return next(iter(resolved)), "verified_dated_alias"
    return None, "unmapped_or_ambiguous_target_issuer"


def build_features(panel, events, etfs, *, peer_panel=None):
    """Return zero-centered feature arrays, a shared ``peer_valid``, and audit.

    Each event preserves the caller's symbol order (49 in production). Peer
    membership and all prices are evaluated at that event, never the label end.
    Select up to ten strictly positive Pearson correlations of adjusted daily
    simple returns, using 252 returns. Ties break lexically by security ID. One
    security per unique issuer is retained and the entire focal issuer removed.

    Matched market controls use ALL data-qualified, currently eligible other
    issuers, without positive-correlation or top-ten selection. If multiple
    classes of an issuer remain, retain the highest finite correlation, then
    lexical security ID (negative correlations allowed; undefined sort last).
    This representation rule is only class deduplication, not a market filter.
    Controls share peer_valid exactly: no control effect on invalid peer cells.

    Both breadth measurements use the SAME peers selected at the current event.
    This means historical breadth is conditional on today's known peer set, not
    a claim that those peers were constituents 63 sessions earlier. ETF targets,
    unmapped targets and fewer than five peers give exact zero and valid=False.
    """
    for key in ("adj", "close"):
        _validate_frame(panel[key], "target " + key)
    if not panel["adj"].index.equals(panel["close"].index) or not panel["adj"].columns.equals(panel["close"].columns):
        raise ValueError("Target adjusted/raw close alignment mismatch")
    peers = load_peer_panel() if peer_panel is None else peer_panel
    for key in ("adj", "close", "eligible"):
        _validate_frame(peers[key], "peer " + key)
        if not peers[key].index.equals(peers["adj"].index) or not peers[key].columns.equals(peers["adj"].columns):
            raise ValueError("Peer price/membership alignment mismatch")
    if any(dtype != bool for dtype in peers["eligible"].dtypes):
        raise ValueError("Peer membership must be explicit booleans, with no missing values")
    dates = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    width = len(dates[0][1]) if dates else 0
    if any(len(symbols) != width for _, symbols in dates):
        raise ValueError("All event widths must match")
    raw = {key: np.zeros((len(dates), width), dtype=float) for key in FEATURE_NAMES}
    raw["peer_valid"] = np.zeros((len(dates), width), dtype=bool)
    reasons = Counter()
    records = []
    universe_counts = []
    etfs = set(etfs) | {"SPY", "QQQ"}
    issuer_values = np.array([peers["issuers"].get(s) for s in peers["adj"].columns], dtype=object)
    issuer_valid = np.array([_valid_issuer(v) for v in issuer_values])
    for event_index, (day, symbols) in enumerate(dates):
        if len(set(symbols)) != width:
            raise ValueError("Duplicate event symbols")
        if day not in panel["adj"].index or day not in peers["adj"].index:
            raise ValueError("Event must be an exact target and peer session: " + str(day))
        pt = peers["adj"].index.get_loc(day)
        tt = panel["adj"].index.get_loc(day)
        local_records = []
        if pt >= PRICE_OBSERVATIONS - 1:
            history = peers["adj"].iloc[pt - PRICE_OBSERVATIONS + 1:pt + 1]
            history_raw = peers["close"].iloc[pt - PRICE_OBSERVATIONS + 1:pt + 1]
            prices = history.to_numpy(dtype=float, copy=True)
            raw_prices = history_raw.to_numpy(dtype=float, copy=True)
            valid = (np.isfinite(prices) & (prices > 0) & np.isfinite(raw_prices) & (raw_prices > 0)).all(axis=0)
            valid &= peers["eligible"].iloc[pt].to_numpy(copy=True) & issuer_valid
            pos = np.flatnonzero(valid)
            peer_prices = prices[:, pos]
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = peer_prices[-252:] / peer_prices[-253:-1] - 1.
            centered = returns - returns.mean(axis=0)
            norms = np.sqrt((centered * centered).sum(axis=0))
            now_above = peer_prices[-1] > peer_prices[-MA_SESSIONS:].mean(axis=0)
            old_above = peer_prices[-1-BROADENING_LAG] > peer_prices[:MA_SESSIONS].mean(axis=0)
            names = peers["adj"].columns[pos].tolist()
            peer_issuers = issuer_values[pos]
        else:
            pos = np.array([], dtype=int)
        universe_counts.append(dict(date=str(day.date()), contemporaneous_members=int(peers["eligible"].iloc[pt].sum()),
                                    price_and_identity_valid_candidates=int(len(pos))))
        for column, symbol in enumerate(symbols):
            rec = dict(symbol=symbol, selected_peers=[])
            def skip(reason):
                rec["reason"] = reason
                reasons[reason] += 1
                local_records.append(rec)
            if symbol in etfs:
                skip("ETF_target_neutral")
                continue
            issuer, identity_basis = _target_issuer(symbol, day, peers)
            if issuer is None:
                skip(identity_basis)
                continue
            rec.update(issuer_id=issuer, identity_basis=identity_basis)
            if symbol not in panel["adj"].columns:
                skip("missing_target_price_column")
                continue
            if tt < CORRELATION_RETURNS or not len(pos):
                skip("insufficient_past_price_history")
                continue
            target = panel["adj"].iloc[tt-CORRELATION_RETURNS:tt+1][symbol].to_numpy(dtype=float, copy=True)
            target_raw = panel["close"].iloc[tt-CORRELATION_RETURNS:tt+1][symbol].to_numpy(dtype=float, copy=True)
            target_sessions = panel["adj"].index[tt-CORRELATION_RETURNS:tt+1]
            if not target_sessions.equals(history.index[-CORRELATION_RETURNS-1:]):
                raise ValueError("Target and peer return sessions differ")
            if not (np.isfinite(target) & (target > 0) & np.isfinite(target_raw) & (target_raw > 0)).all():
                skip("invalid_target_price_window")
                continue
            target_returns = target[1:] / target[:-1] - 1.
            target_centered = target_returns - target_returns.mean()
            target_norm = np.sqrt((target_centered * target_centered).sum())
            if not np.isfinite(target_norm) or target_norm <= 1e-14:
                skip("constant_target_returns")
                continue
            denominator = norms * target_norm
            # Explicit reduction avoids the platform BLAS matmul warning path
            # and is inexpensive for the small historical peer cross-section.
            correlations = np.divide((target_centered[:, None] * centered).sum(axis=0), denominator,
                                     out=np.full(len(names), np.nan), where=denominator > 1e-14)
            acceptable = [i for i in range(len(names)) if peer_issuers[i] != issuer and
                          np.isfinite(correlations[i]) and correlations[i] > 0]
            acceptable.sort(key=lambda i: (-float(correlations[i]), names[i]))
            chosen, seen_issuers = [], set()
            for i in acceptable:
                if peer_issuers[i] in seen_issuers:
                    continue
                seen_issuers.add(peer_issuers[i])
                chosen.append(i)
                if len(chosen) == MAX_PEERS:
                    break
            rec["positive_correlation_unique_issuers"] = len(set(peer_issuers[i] for i in acceptable))
            rec["selected_peers"] = [dict(security_id=names[i], issuer_id=peer_issuers[i],
                                           correlation=float(correlations[i])) for i in chosen]
            if len(chosen) < MIN_PEERS:
                skip("fewer_than_five_positive_correlation_issuers")
                continue
            current = float(now_above[chosen].mean())
            previous = float(old_above[chosen].mean())
            raw["peer_participation"][event_index, column] = current - .5
            raw["peer_broadening"][event_index, column] = (current - previous) / 2.
            # Matched controls keep identical target validity and the full
            # price-qualified historical universe, but remove the correlated
            # top-ten selection. Membership and breadth dates are unchanged.
            market_candidates = [i for i in range(len(names)) if peer_issuers[i] != issuer]
            market_candidates.sort(key=lambda i: (-float(correlations[i]) if np.isfinite(correlations[i])
                                                   else float("inf"), names[i]))
            market_chosen, market_issuers = [], set()
            for i in market_candidates:
                if peer_issuers[i] not in market_issuers:
                    market_issuers.add(peer_issuers[i])
                    market_chosen.append(i)
            market_current = float(now_above[market_chosen].mean())
            market_previous = float(old_above[market_chosen].mean())
            raw["market_participation"][event_index, column] = market_current - .5
            raw["market_broadening"][event_index, column] = (market_current - market_previous) / 2.
            raw["peer_valid"][event_index, column] = True
            rec.update(reason="valid", above_MA200_now=current, above_MA200_63_sessions_ago=previous,
                       market_above_MA200_now=market_current, market_above_MA200_63_sessions_ago=market_previous,
                       market_distinct_other_issuers=len(market_chosen),
                       market_representatives=[dict(security_id=names[i], issuer_id=peer_issuers[i]) for i in market_chosen])
            local_records.append(rec)
        records.append(dict(date=str(day.date()), targets=local_records))
    audit = dict(events=len(dates), symbols_per_event=width,
                 feature_names=list(FEATURE_NAMES), valid_cells=int(raw["peer_valid"].sum()),
                 missing_or_neutral_reasons=dict(reasons), max_peers=MAX_PEERS, min_peers=MIN_PEERS,
                 correlation_returns=CORRELATION_RETURNS, correlation_basis="adjusted-close simple daily returns",
                 positive_raw_and_adjusted_price_observations=PRICE_OBSERVATIONS,
                 MA_sessions=MA_SESSIONS, broadening_lag_sessions=BROADENING_LAG,
                 score_bounds=[-.5, .5], feature_horizon="current context; core skip42 momentum unchanged",
                 same_current_peer_set_at_both_breadth_dates=True,
                 peer_membership_at_old_breadth_date_required=False,
                 entire_focal_issuer_excluded=True, unique_peer_issuers=True,
                 ties="descending correlation then lexical security ID; one per issuer",
                 matched_market_controls="All current data-qualified other issuers, no positive-correlation gate or top10; exactly the same peer_valid target mask",
                 market_class_representative="highest finite correlation (negative allowed), then lexical security ID; undefined correlations last",
                 missing_policy="exact zero correction plus false peer_valid; trading eligibility untouched",
                 ETF_targets="neutral; no look-through holdings assumed",
                 source_scope=peers.get("source_scope", "injected_test_peer_panel"),
                 strict_complete_history_certified=bool(peers.get("strict_complete_history_certified", False)),
                 source_hashes=dict(peers.get("source_hashes", {})),
                 universe_by_event=universe_counts, targets_by_event=records,
                 limitations=["Research membership and issuer continuity are not complete vendor-certified history.",
                              "Available-data exclusions and archival completeness can introduce selection bias.",
                              "Price correlation does not identify economic industry membership.",
                              "Broadening uses the peer set known at the current signal, not future membership."])
    return raw, audit
