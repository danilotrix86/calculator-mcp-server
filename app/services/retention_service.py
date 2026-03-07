from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
from collections import defaultdict

from app.services.supabase_service import get_supabase_service
from fastapi.concurrency import run_in_threadpool


def _parse_dt(dt_str: str) -> datetime:
    """Parse an ISO timestamp from Supabase into a naive UTC datetime."""
    if "." in dt_str:
        dt_str = dt_str.split(".")[0]
    else:
        dt_str = dt_str.split("+")[0].replace("Z", "")
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")


def _week_start(dt: datetime) -> str:
    """Return the Monday of the week containing *dt* as YYYY-MM-DD."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


async def get_retention_metrics() -> Dict[str, Any]:
    """Compute all retention analytics in a single call.

    Returns KPIs, cohort retention, DAU/WAU time series, user segmentation,
    and query frequency distribution.
    """
    svc = get_supabase_service()
    if not svc.client:
        return _empty_response()

    try:
        users = await _fetch_all_users(svc)
        queries = await _fetch_all_queries(svc)
    except Exception as e:
        logging.error("retention_service: error fetching data: %s", e)
        return _empty_response()

    now = datetime.utcnow()

    user_signup: Dict[str, datetime] = {}
    for u in users:
        uid = u["id"]
        user_signup[uid] = _parse_dt(u["created_at"])

    query_events: List[Dict[str, Any]] = []
    for q in queries:
        uid = q.get("user_id")
        if not uid:
            continue
        query_events.append({
            "user_id": uid,
            "created_at": _parse_dt(q["created_at"]),
        })

    kpis = _compute_kpis(user_signup, query_events, now)
    cohort_table = _compute_cohort_retention(user_signup, query_events, now)
    dau_series = _compute_dau_series(query_events, user_signup, now)
    segmentation = _compute_user_segmentation(query_events, user_signup, now)
    frequency = _compute_frequency_distribution(query_events, now)

    return {
        "kpis": kpis,
        "cohortTable": cohort_table,
        "dauSeries": dau_series,
        "segmentation": segmentation,
        "frequencyDistribution": frequency,
    }


def _empty_response() -> Dict[str, Any]:
    return {
        "kpis": {},
        "cohortTable": [],
        "dauSeries": [],
        "segmentation": [],
        "frequencyDistribution": [],
    }


async def _fetch_all_users(svc) -> List[Dict[str, Any]]:
    """Fetch all users from app_users (id, created_at)."""
    page_size = 1000
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        q = (
            svc.client.table("app_users")
            .select("id, created_at")
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
        )
        result = await run_in_threadpool(q.execute)
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


async def _fetch_all_queries(svc) -> List[Dict[str, Any]]:
    """Fetch all non-cache queries (user_id, created_at only)."""
    page_size = 1000
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        q = (
            svc.client.table("user_queries")
            .select("user_id, created_at")
            .neq("query_type", "image_hash_cache")
            .not_.is_("user_id", "null")
            .order("created_at", desc=False)
            .range(offset, offset + page_size - 1)
        )
        result = await run_in_threadpool(q.execute)
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


# ──────────────────────────── KPIs ────────────────────────────

def _compute_kpis(
    user_signup: Dict[str, datetime],
    query_events: List[Dict[str, Any]],
    now: datetime,
) -> Dict[str, Any]:
    total_users = len(user_signup)

    active_7d = set()
    active_30d = set()
    new_7d = 0
    new_30d = 0
    queries_30d = 0

    cutoff_7 = now - timedelta(days=7)
    cutoff_30 = now - timedelta(days=30)

    for uid, signup in user_signup.items():
        if signup >= cutoff_7:
            new_7d += 1
        if signup >= cutoff_30:
            new_30d += 1

    for ev in query_events:
        dt = ev["created_at"]
        uid = ev["user_id"]
        if dt >= cutoff_7:
            active_7d.add(uid)
        if dt >= cutoff_30:
            active_30d.add(uid)
            queries_30d += 1

    avg_queries = round(queries_30d / len(active_30d), 1) if active_30d else 0

    # D1 / D7 / D30 retention: % of users who signed up in the last 60 days
    # and had activity after their signup day / week / month.
    d1_ret = _day_n_retention(user_signup, query_events, now, 1)
    d7_ret = _day_n_retention(user_signup, query_events, now, 7)
    d30_ret = _day_n_retention(user_signup, query_events, now, 30)

    return {
        "totalUsers": total_users,
        "activeUsers7d": len(active_7d),
        "activeUsers30d": len(active_30d),
        "newUsers7d": new_7d,
        "newUsers30d": new_30d,
        "avgQueriesPerUser30d": avg_queries,
        "retentionD1": d1_ret,
        "retentionD7": d7_ret,
        "retentionD30": d30_ret,
    }


def _day_n_retention(
    user_signup: Dict[str, datetime],
    query_events: List[Dict[str, Any]],
    now: datetime,
    n: int,
) -> float:
    """Percentage of users (who signed up >=n days ago in the last 90 days)
    that had at least one query on or after day n from signup."""
    cutoff_early = now - timedelta(days=90)
    cutoff_late = now - timedelta(days=n)
    eligible: set[str] = set()
    for uid, signup in user_signup.items():
        if cutoff_early <= signup <= cutoff_late:
            eligible.add(uid)
    if not eligible:
        return 0.0

    user_query_dates: Dict[str, datetime] = {}
    for ev in query_events:
        uid = ev["user_id"]
        if uid in eligible:
            dt = ev["created_at"]
            if uid not in user_query_dates or dt > user_query_dates[uid]:
                user_query_dates[uid] = dt

    retained = 0
    for uid in eligible:
        signup = user_signup[uid]
        threshold = signup + timedelta(days=n)
        latest_query = user_query_dates.get(uid)
        if latest_query and latest_query >= threshold:
            retained += 1

    return round(retained / len(eligible) * 100, 1)


# ──────────────────────── Cohort retention ────────────────────────

def _compute_cohort_retention(
    user_signup: Dict[str, datetime],
    query_events: List[Dict[str, Any]],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Weekly cohort retention table (last 12 weeks)."""
    cohorts: Dict[str, set] = defaultdict(set)
    for uid, signup in user_signup.items():
        week = _week_start(signup)
        cohorts[week].add(uid)

    user_active_weeks: Dict[str, set] = defaultdict(set)
    for ev in query_events:
        uid = ev["user_id"]
        week = _week_start(ev["created_at"])
        user_active_weeks[uid].add(week)

    sorted_weeks = sorted(cohorts.keys(), reverse=True)[:12]
    sorted_weeks.reverse()

    table = []
    for cohort_week in sorted_weeks:
        cohort_users = cohorts[cohort_week]
        cohort_size = len(cohort_users)
        if cohort_size == 0:
            continue

        cohort_monday = datetime.strptime(cohort_week, "%Y-%m-%d")
        max_weeks = min(8, max(0, (now - cohort_monday).days // 7))

        retention_pcts: List[float | None] = []
        for w in range(max_weeks + 1):
            target_monday = cohort_monday + timedelta(weeks=w)
            target_week = target_monday.strftime("%Y-%m-%d")
            active_count = sum(
                1 for uid in cohort_users
                if target_week in user_active_weeks.get(uid, set())
            )
            retention_pcts.append(round(active_count / cohort_size * 100, 1))

        table.append({
            "cohortWeek": cohort_week,
            "cohortSize": cohort_size,
            "retention": retention_pcts,
        })

    return table


# ──────────────────────── DAU / WAU series ────────────────────────

def _compute_dau_series(
    query_events: List[Dict[str, Any]],
    user_signup: Dict[str, datetime],
    now: datetime,
) -> List[Dict[str, Any]]:
    """DAU and new-user registrations per day (last 60 days)."""
    cutoff = now - timedelta(days=60)

    daily_active: Dict[str, set] = defaultdict(set)
    for ev in query_events:
        dt = ev["created_at"]
        if dt >= cutoff:
            day_key = dt.strftime("%Y-%m-%d")
            daily_active[day_key].add(ev["user_id"])

    daily_new: Dict[str, int] = defaultdict(int)
    for uid, signup in user_signup.items():
        if signup >= cutoff:
            day_key = signup.strftime("%Y-%m-%d")
            daily_new[day_key] += 1

    all_days = set(daily_active.keys()) | set(daily_new.keys())
    series = []
    for day in sorted(all_days):
        series.append({
            "date": day,
            "dau": len(daily_active.get(day, set())),
            "newUsers": daily_new.get(day, 0),
        })

    return series


# ──────────────────────── User segmentation ────────────────────────

def _compute_user_segmentation(
    query_events: List[Dict[str, Any]],
    user_signup: Dict[str, datetime],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Segment users into power / regular / occasional / inactive."""
    cutoff_30 = now - timedelta(days=30)
    cutoff_7 = now - timedelta(days=7)
    weeks_in_range = max(1, 30 / 7)

    queries_per_user: Dict[str, int] = defaultdict(int)
    for ev in query_events:
        if ev["created_at"] >= cutoff_30:
            queries_per_user[ev["user_id"]] += 1

    power = 0
    regular = 0
    occasional = 0
    inactive = 0

    for uid in user_signup:
        count = queries_per_user.get(uid, 0)
        weekly_avg = count / weeks_in_range
        if weekly_avg >= 5:
            power += 1
        elif weekly_avg >= 2:
            regular += 1
        elif count >= 1:
            occasional += 1
        else:
            inactive += 1

    return [
        {"segment": "Power Users (5+/week)", "count": power},
        {"segment": "Regular (2-4/week)", "count": regular},
        {"segment": "Occasional (1/week)", "count": occasional},
        {"segment": "Inactive (0 in 30d)", "count": inactive},
    ]


# ──────────────────── Query frequency distribution ────────────────────

def _compute_frequency_distribution(
    query_events: List[Dict[str, Any]],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Histogram of query counts per user (last 30 days)."""
    cutoff = now - timedelta(days=30)
    per_user: Dict[str, int] = defaultdict(int)
    for ev in query_events:
        if ev["created_at"] >= cutoff:
            per_user[ev["user_id"]] += 1

    buckets = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51+", 51, 999999),
    ]

    dist = []
    for label, lo, hi in buckets:
        count = sum(1 for c in per_user.values() if lo <= c <= hi)
        dist.append({"bucket": label, "users": count})

    return dist
