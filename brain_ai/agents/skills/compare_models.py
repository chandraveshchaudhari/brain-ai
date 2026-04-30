from typing import Any, Dict, List


def compare_models(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a simple leaderboard from multiple experiment results.

    Expects `results` to be a list of dicts with a `metrics` key.
    """
    leaderboard = []
    for r in results:
        payload = r.get("result", r)
        metrics = payload.get("metrics", {})
        score = metrics.get("score", metrics.get("rmse", None))
        leaderboard.append({"result": r, "score": score})
    leaderboard.sort(key=lambda x: x["score"] if x["score"] is not None else float("inf"))
    return {"status": "ok", "leaderboard": leaderboard}
