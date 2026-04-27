import json
import math
import pickle

from .recommender import Recommender


class AdaptiveHybridRecommender(Recommender):
    SOURCE_HSTU = "hstu"
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
        max_hstu=30,
        min_prev_time_for_override=0.35,
        margin=0.02,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.catalog = catalog
        self.baseline_recommender = baseline_recommender
        self.fallback_recommender = fallback_recommender

        self.max_history = int(max_history)
        self.max_hstu = int(max_hstu)
        self.min_prev_time_for_override = float(min_prev_time_for_override)
        self.margin = float(margin)

        self._hstu_cache = {}

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

    def _load_hstu(self, user):
        user = int(user)

        if user in self._hstu_cache:
            return self._hstu_cache[user]

        raw = self.hstu_redis.get(user)
        recs = self._loads_pickle(raw, [])

        result = []

        for x in recs:
            value = self._safe_int(x)
            if value is not None:
                result.append(value)

        self._hstu_cache[user] = result
        return result

    def _bandit_key(self, user):
        return f"user:{int(user)}:adaptive_hybrid_fast:source_score"

    def _pending_key(self, user):
        return f"user:{int(user)}:adaptive_hybrid_fast:pending"

    def _source_multiplier(self, user, source):
        raw = self.listen_history_redis.hget(self._bandit_key(user), source)

        if raw is None:
            return 1.0

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            ema = float(raw)
            return min(1.25, max(0.80, 0.80 + ema))
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

        new = 0.88 * old + 0.12 * reward
        self.listen_history_redis.hset(key, source, new)

    def _remember_recommendation_source(self, user, track, source):
        payload = json.dumps(
            {
                "track": int(track),
                "source": str(source),
            }
        )

        self.listen_history_redis.setex(self._pending_key(user), 3600, payload)

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

    def _best_hstu_candidate(self, user, seen):
        best = None
        best_score = -1.0

        hstu_recs = self._load_hstu(user)

        for rank, cand in enumerate(hstu_recs[: self.max_hstu], start=1):
            cand = self._safe_int(cand)

            if cand is None:
                continue

            if cand in seen:
                continue

            score = 1.0 / math.sqrt(rank + 1.0)
            score *= self._source_multiplier(user, self.SOURCE_HSTU)

            if score > best_score:
                best = cand
                best_score = score

        return best, best_score

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        user = int(user)
        prev_track = int(prev_track)
        prev_track_time = float(prev_track_time)

        self._update_bandit_from_previous_result(user, prev_track, prev_track_time)

        baseline = self._safe_baseline(user, prev_track, prev_track_time)
        history = self._load_history(user)
        seen = {int(track) for track, _ in history}

        hstu_candidate, hstu_score = self._best_hstu_candidate(user, seen)

        if hstu_candidate is None:
            if baseline is not None:
                self._remember_recommendation_source(user, baseline, self.SOURCE_BASELINE)
                return baseline

            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        use_hstu = False

        if baseline is None:
            use_hstu = True
        elif hstu_candidate != baseline:
            if prev_track_time >= self.min_prev_time_for_override:
                baseline_score = 0.48 * self._source_multiplier(user, self.SOURCE_BASELINE)

                if hstu_score >= baseline_score + self.margin:
                    use_hstu = True

        if use_hstu:
            self._remember_recommendation_source(user, hstu_candidate, self.SOURCE_HSTU)
            return int(hstu_candidate)

        self._remember_recommendation_source(user, baseline, self.SOURCE_BASELINE)
        return int(baseline)
