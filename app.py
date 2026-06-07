import json
import logging
import os
import uuid
from datetime import datetime

from flask import Flask, current_app, jsonify, render_template, request

from config.settings import (
    DEBUG,
    SECRET_KEY,
    SESSION_DIR,
)
from data.loader import load_all_cities
from data.preprocessor import preprocess_city
from engine.clustering import cluster_city
from engine.cross_city import recommend_cross_city
from engine.recommender import recommend
from engine.similarity import build_index
from nlp.gemini import create_client, extract_features, generate_explanation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _new_session() -> dict:
    return {
        "session_id":        str(uuid.uuid4()),
        "created_at":        datetime.utcnow().isoformat(),
        "last_accessed":     datetime.utcnow().isoformat(),
        "messages":          [],
        "city":              None,
        "listing_scenario":  None,
        "expected_price":    None,
        "last_features":     None,
        "detected_language": "he",
    }


# Fields that must be non-null before the ML pipeline runs.
_REQUIRED_NEW = frozenset({
    "neighbourhood_cleansed", "property_type", "room_type",
    "accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
    "has_wifi", "has_kitchen", "has_tv", "has_ac",
    "has_pool", "has_parking", "has_washer", "has_elevator",
})

_REQUIRED_EXISTING_EXTRA = frozenset({
    "review_scores_rating", "review_scores_location", "review_scores_value",
    "host_is_superhost", "instant_bookable",
})


def _is_ready_server(session: dict) -> bool:
    """Return True only when all required fields for the detected scenario are present.

    This is the authoritative readiness check — it overrides Gemini's ready_to_recommend
    flag, which is advisory only.
    """
    scenario = session.get("listing_scenario")
    if not scenario:
        return False
    if not session.get("city"):
        return False
    features = session.get("last_features") or {}
    if not all(features.get(f) is not None for f in _REQUIRED_NEW):
        return False
    if scenario == "existing_listing":
        if not all(features.get(f) is not None for f in _REQUIRED_EXISTING_EXTRA):
            return False
        if session.get("expected_price") is None:
            return False
    return True


def _read_session(session_id: str) -> dict:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return _new_session()
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Corrupt session '%s' — starting fresh.", session_id)
        return _new_session()


def _write_session(session: dict) -> None:
    session["last_accessed"] = datetime.utcnow().isoformat()
    sid = session["session_id"]
    tmp    = SESSION_DIR / f"{sid}.tmp"
    target = SESSION_DIR / f"{sid}.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False)
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# City initialisation
# ---------------------------------------------------------------------------

def _init_cities() -> dict:
    """Load, preprocess, cluster, and index all cities. Called once at startup."""
    raw = load_all_cities()
    cities = {}
    for name, df in raw.items():
        logger.info("Initialising city '%s' ...", name)
        city_data = preprocess_city(df)
        city_data = cluster_city(city_data, name)
        idx       = build_index(city_data)
        cities[name] = {"data": city_data, "index": idx}
        logger.info("City '%s' ready.", name)
    return cities


# ---------------------------------------------------------------------------
# Flask factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["DEBUG"]      = DEBUG

    logger.info("Loading city data ...")
    app.config["CITIES"] = _init_cities()
    logger.info("All cities ready: %s", sorted(app.config["CITIES"]))

    app.config["GEMINI_CLIENT"] = create_client()

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/chat", methods=["POST"])
    def chat():
        body         = request.get_json(silent=True) or {}
        user_message = (body.get("message") or "").strip()
        session_id   = (body.get("session_id") or "").strip()

        if not user_message:
            return jsonify({"error": "message is required"}), 400

        session = _read_session(session_id) if session_id else _new_session()
        cities  = current_app.config["CITIES"]
        client  = current_app.config["GEMINI_CLIENT"]
        city_names = sorted(cities.keys())

        # ── 1. Domain check + feature extraction (single Gemini call) ─────────
        extracted, error_reply = extract_features(
            client, city_names, session["messages"], user_message,
            known_session=session,
        )
        if extracted is None:
            _append_and_save(session, user_message, error_reply)
            return jsonify({"reply": error_reply, "session_id": session["session_id"]})

        # ── 2. Accumulate partial features across turns ───────────────────────
        if extracted.get("features"):
            prev  = session.get("last_features") or {}
            fresh = {k: v for k, v in extracted["features"].items() if v is not None}
            session["last_features"] = {**prev, **fresh}
        if extracted.get("city"):
            session["city"] = extracted["city"].lower()
        if extracted.get("listing_scenario"):
            session["listing_scenario"] = extracted["listing_scenario"]
        if extracted.get("expected_price") is not None:
            session["expected_price"] = extracted["expected_price"]

        # Language is sticky: once Hebrew is detected it never reverts to English.
        # This prevents short numeric answers from flipping the conversation language.
        new_lang = extracted.get("detected_language")
        if new_lang == "he":
            session["detected_language"] = "he"
        elif new_lang == "en" and session.get("detected_language") != "he":
            session["detected_language"] = "en"

        # ── 3. Ask follow-up if not enough info yet ───────────────────────────
        # _is_ready_server is the authoritative gate; Gemini's flag is advisory only.
        if not _is_ready_server(session):
            reply = extracted.get("follow_up") or "Could you tell me more about your property?"
            _append_and_save(session, user_message, reply)
            return jsonify({"reply": reply, "session_id": session["session_id"]})

        # ── 4. City validation / cross-city fallback ──────────────────────────
        city = (session.get("city") or "").lower()
        if city not in cities:
            features        = session["last_features"] or {}
            cross_result    = recommend_cross_city(features, city or "unknown", cities)
            explanation     = generate_explanation(
                client, city or "unknown", features, cross_result.rec,
                session.get("detected_language", "he"),
                is_cross_city=True,
                source_cities=cross_result.source_cities,
                listing_scenario=session.get("listing_scenario"),
                expected_price=session.get("expected_price"),
            )
            _append_and_save(session, user_message, explanation)
            return jsonify({
                "reply":      explanation,
                "session_id": session["session_id"],
                "recommendation": {
                    "recommended_price": cross_result.rec.recommended_price,
                    "price_min":         cross_result.rec.price_min,
                    "price_max":         cross_result.rec.price_max,
                    "tier_label":        cross_result.rec.tier_label,
                    "confidence_pct":    cross_result.rec.confidence_pct,
                    "comparables":       cross_result.rec.comparables,
                    "is_cross_city":     True,
                    "source_cities":     cross_result.source_cities,
                },
            })

        # ── 5. Run ML pipeline ────────────────────────────────────────────────
        features       = session["last_features"] or {}
        expected_price = session.get("expected_price")
        city_state     = cities[city]

        try:
            rec = recommend(
                query_features=features,
                city_name=city,
                city_data=city_state["data"],
                index=city_state["index"],
                expected_price=expected_price,
            )
        except Exception:
            logger.exception("Recommendation failed for city '%s'.", city)
            reply = "I ran into a problem computing the recommendation. Please try again."
            _append_and_save(session, user_message, reply)
            return jsonify({"reply": reply, "session_id": session["session_id"]})

        # ── 6. Generate natural-language explanation (second Gemini call) ─────
        explanation = generate_explanation(
            client, city, features, rec,
            session.get("detected_language", "he"),
            listing_scenario=session.get("listing_scenario"),
            expected_price=session.get("expected_price"),
        )

        # ── 7. Persist and respond ────────────────────────────────────────────
        _append_and_save(session, user_message, explanation)

        return jsonify({
            "reply":      explanation,
            "session_id": session["session_id"],
            "recommendation": {
                "recommended_price": rec.recommended_price,
                "price_min":         rec.price_min,
                "price_max":         rec.price_max,
                "tier_label":        rec.tier_label,
                "confidence_pct":    rec.confidence_pct,
                "comparables":       rec.comparables,
                "is_cross_city":     False,
                "source_cities":     [],
            },
        })


def _append_and_save(session: dict, user_msg: str, assistant_msg: str) -> None:
    session["messages"].append({"role": "user",      "content": user_msg})
    session["messages"].append({"role": "assistant", "content": assistant_msg})
    _write_session(session)


# ---------------------------------------------------------------------------
# Module-level app instance — required by gunicorn (gunicorn app:app)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=DEBUG, port=5000)
