import json
import math
import pickle
from collections import defaultdict

from .recommender import Recommender


class AdaptiveHybridRecommender(Recommender):
    SOURCE_HSTU = "hstu"
    SOURCE_SASREC = "sasrec"
    SOURCE_LFM = "lfm"
    SOURCE_BASELINE = "baseline"

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        sasrec_redis,
        lightfm_redis,
        tracks_redis,
        catalog,
        baseline_recommender,
        fallback_recommender,
        max_history=10,
        max_hstu=80,
        topk_per_anchor=25,
        min_prev_time_for_override=0.42,
        margin=0.055,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.tracks_redis = tracks_redis
        self.catalog = catalog
        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.max_history = int(max_history)
        self.max_hstu = int(max_hstu)
        self.topk_per_anchor = int(topk_per_anchor)
        self.min_prev_time_for_override = float(min_prev_time_for_override)
        self.margin = float(margin)

        self._i2i_cache = {}
        self._track_cache = {}

    def _safe_int(self, value, default=None):
        try:
            return int(value)
        except Exception:
            return default

    def _loads_pickle(self, raw, default):
        if raw is None:
            return default

        try:
            return pickle.loads(raw)
        except Exception:
            pass

        try:
            return self.catalog.from_bytes(raw)
        except Exception:
            return default

    def _load_history(self, user):
        key = f"user:{int(user)}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, self.max_history - 1)

        history = []

        for raw in raw_entries:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                row = json.loads(raw)
                history.append((int(row["track"]), float(row["time"])))
            except Exception:
                continue

        return history

    def _load_i2i(self, source, track):
        track = int(track)
        cache_key = (source, track)

        if cache_key in self._i2i_cache:
            return self._i2i_cache[cache_key]

        if source == self.SOURCE_SASREC:
            redis_conn = self.sasrec_redis
        else:
            redis_conn = self.lightfm_redis

        raw = redis_conn.get(track)
        recs = self._loads_pickle(raw, [])

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        self._i2i_cache[cache_key] = result
        return result

    def _load_hstu(self, user):
        raw = self.hstu_redis.get(int(user))
        recs = self._loads_pickle(raw, [])

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        return result

    def _track_info(self, track):
        track = int(track)

        if track in self._track_cache:
            return self._track_cache[track]

        raw = self.tracks_redis.get(track)
        info = self._loads_pickle(raw, None)
        self._track_cache[track] = info

        return info

    def _artist(self, track):
        info = self._track_info(track)

        if info is None:
            return None

        if isinstance(info, dict):
            return info.get("artist")

        return getattr(info, "artist", None)

    def _bandit_key(self, user):
        return f"user:{int(user)}:adaptive_hybrid:source_score"

    def _pending_key(self, user):
        return f"user:{int(user)}:adaptive_hybrid:pending"

    def _source_multiplier(self, user, source):
        raw = self.listen_history_redis.hget(self._bandit_key(user), source)

        if raw is None:
            return 1.0

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            ema = float(raw)
            return min(1.35, max(0.72, 0.75 + ema))
        except Exception:
            return 1.0

    def _update_bandit_from_previous_result(self, user, prev_track, prev_track_time):
        raw = self.listen_history_redis.get(self._pending_key(user))

        if raw is None:
            return

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            pending = json.loads(raw)

            if int(pending.get("track")) != int(prev_track):
                return

            source = str(pending.get("source"))
        except Exception:
            return

        reward = max(0.0, min(1.0, float(prev_track_time)))

        key = self._bandit_key(user)
        old_raw = self.listen_history_redis.hget(key, source)

        try:
            if isinstance(old_raw, bytes):
                old_raw = old_raw.decode("utf-8")

            old = float(old_raw) if old_raw is not None else 0.50
        except Exception:
            old = 0.50

        new = 0.82 * old + 0.18 * reward
        self.listen_history_redis.hset(key, source, new)

    def _remember_recommendation_source(self, user, track, source):
        payload = json.dumps(
            {
                "track": int(track),
                "source": str(source),
            }
        )

        self.listen_history_redis.setex(self._pending_key(user), 3600, payload)

    def _add_candidate(self, table, cand, source, value):
        cand = self._safe_int(cand)

        if cand is None:
            return

        row = table[cand]
        row["score"] += float(value)
        row["sources"][source] += float(value)

    def _build_candidates(self, user, history, seen, baseline):
        table = defaultdict(lambda: {"score": 0.0, "sources": defaultdict(float)})

        if baseline is not None and baseline not in seen:
            self._add_candidate(table, baseline, self.SOURCE_BASELINE, 1.10)

        hstu_recs = self._load_hstu(user)

        for rank, cand in enumerate(hstu_recs[: self.max_hstu], start=1):
            if cand in seen:
                continue

            value = 1.0 / math.sqrt(rank + 2.0)
            self._add_candidate(table, cand, self.SOURCE_HSTU, value)

        recent = history[: self.max_history]

        for pos, pair in enumerate(recent):
            anchor, listened_time = pair

            listened_time = max(0.0, min(1.0, float(listened_time)))
            anchor_weight = (0.72 ** pos) * (0.30 + listened_time)

            for source in (self.SOURCE_SASREC, self.SOURCE_LFM):
                neighbours = self._load_i2i(source, anchor)[: self.topk_per_anchor]

                for rank, cand in enumerate(neighbours, start=1):
                    if cand in seen:
                        continue

                    value = anchor_weight / (rank + 1.5)

                    if source == self.SOURCE_SASREC:
                        value *= 0.95
                    else:
                        value *= 0.80

                    self._add_candidate(table, cand, source, value)

        for cand, row in table.items():
            adjusted = 0.0

            for source, value in row["sources"].items():
                adjusted += value * self._source_multiplier(user, source)

            row["score"] = adjusted

        for bad in list(table.keys()):
            if bad in seen:
                del table[bad]

        return table

    def _repetition_penalty(self, cand, history):
        cand_artist = self._artist(cand)

        if cand_artist is None:
            return 0.0

        recent_artists = []

        for track, _ in history[:4]:
            recent_artists.append(self._artist(track))

        repeats = sum(1 for artist in recent_artists if artist == cand_artist)

        if repeats <= 1:
            return 0.0

        return 0.10 * (repeats - 1)

    def _choose_source_label(self, row):
        if not row["sources"]:
            return "unknown"

        return max(row["sources"].items(), key=lambda x: x[1])[0]

    def _safe_baseline(self, user, prev_track, prev_track_time):
        try:
            rec = self.baseline_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)
        except Exception:
            pass

        try:
            rec = self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
            return int(rec)
        except Exception:
            return None

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        user = int(user)
        prev_track = int(prev_track)
        prev_track_time = float(prev_track_time)

        self._update_bandit_from_previous_result(user, prev_track, prev_track_time)

        baseline = self._safe_baseline(user, prev_track, prev_track_time)
        history = self._load_history(user)
        seen = {int(track) for track, _ in history}

        if not history:
            if baseline is not None:
                self._remember_recommendation_source(user, baseline, self.SOURCE_BASELINE)
                return baseline

            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        candidates = self._build_candidates(user, history, seen, baseline)

        if not candidates:
            if baseline is not None:
                self._remember_recommendation_source(user, baseline, self.SOURCE_BASELINE)
                return baseline

            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        ranked = []

        for cand, row in candidates.items():
            score = row["score"] - self._repetition_penalty(cand, history)
            ranked.append((score, int(cand), row))

        ranked.sort(reverse=True)

        best_score, best, best_row = ranked[0]

        baseline_score = None

        if baseline in candidates:
            baseline_score = candidates[baseline]["score"] - self._repetition_penalty(
                baseline, history
            )

        use_best = baseline is None

        if baseline is not None and best != baseline:
            if prev_track_time >= self.min_prev_time_for_override:
                if baseline_score is None or best_score >= baseline_score + self.margin:
                    use_best = True

        if use_best:
            source = self._choose_source_label(best_row)
            self._remember_recommendation_source(user, best, source)
            return int(best)

        self._remember_recommendation_source(user, baseline, self.SOURCE_BASELINE)
        return int(baseline)
