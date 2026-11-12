"""DS-341 Customer Analytics dashboard. Reads results/*.json written by run_pipeline.py."""
import json
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, render_template

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
app = Flask(__name__)


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        abort(503, description=f"results/{name}.json is missing. Run: python run_pipeline.py")
    return json.loads(path.read_text(encoding="utf-8"))


def context():
    return {"overview": load("overview"), "segments": load("segments"), "basket": load("basket"), "churn": load("churn")}


@app.route("/")
def overview():
    return render_template("overview.html", **context())


@app.route("/segments")
def segments():
    return render_template("segments.html", **context())


@app.route("/basket")
def basket():
    return render_template("basket.html", **context())


@app.route("/churn")
def churn():
    return render_template("churn.html", **context())


@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html", **context())


@app.route("/api/<name>")
def api(name):
    if name not in {"overview", "segments", "basket", "churn"}:
        abort(404)
    return jsonify(load(name))


@app.errorhandler(404)
def not_found(_):
    return render_template("message.html", title="Not found", message="That page does not exist."), 404


@app.errorhandler(503)
def missing(err):
    return render_template("message.html", title="Results missing", message=err.description), 503


if __name__ == "__main__":
    app.run(debug=True, port=5001)
