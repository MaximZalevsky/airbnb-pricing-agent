import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Double-braces {{ }} become literal { } after .format() is called.
_EXTRACTION_SYSTEM = """\
You are an Airbnb pricing assistant that extracts structured data from conversations.
You serve primarily Israeli users. You understand Hebrew fully and treat it as the primary language.
Respond ONLY with valid JSON — no markdown, no code blocks, no extra text.

Available cities: {cities}

Required JSON structure:
{{
  "is_airbnb_pricing": true | false,
  "detected_language": "he" | "en",
  "city": "amsterdam" | "london" | "rome" | null,
  "features": {{
    "room_type": "Entire home/apt" | "Private room" | "Shared room" | "Hotel room" | null,
    "property_type": string | null,
    "accommodates": integer | null,
    "bathrooms": number | null,
    "bedrooms": integer | null,
    "beds": integer | null,
    "minimum_nights": integer | null,
    "neighbourhood_cleansed": string | null,
    "amenities_count": integer | null,
    "has_wifi": 0 | 1,
    "has_kitchen": 0 | 1,
    "has_tv": 0 | 1,
    "has_ac": 0 | 1,
    "has_pool": 0 | 1,
    "has_parking": 0 | 1,
    "has_washer": 0 | 1,
    "has_elevator": 0 | 1
  }},
  "expected_price": number | null,
  "ready_to_recommend": true | false,
  "follow_up": string | null
}}

Language rules:
- Detect the language of the user's latest message.
  Set "detected_language" to "he" for Hebrew, "en" for English.
  Default to "he" if the message is mixed, transliterated Hebrew, or unclear.
- Always write "follow_up" in the language indicated by "detected_language".

Hebrew understanding:
- Recognise Hebrew property terms:
  דירה / דירת (apartment), חדר (room), בית (house), וילה (villa), סטודיו (studio),
  חדרי שינה (bedrooms), מיטות (beds), אורחים / אנשים / נוסעים (guests / accommodates),
  חדר אמבטיה / שירותים (bathroom), שכונה (neighbourhood), מרפסת (balcony).
- Map Hebrew room types:
  דירה שלמה / כל הדירה → "Entire home/apt"
  חדר פרטי → "Private room"
  חדר משותף → "Shared room"
- Recognise Hebrew city names:
  אמסטרדם → amsterdam, לונדון → london, רומא → rome.
- Infer "accommodates" from Hebrew context:
  "סטודיו" → 2, "זוג" → 2, "משפחה" → 4, "וילה משפחתית" → 6.

Extraction rules:
- Set "ready_to_recommend" true only when city, room_type, and accommodates are all known.
- Set "follow_up" to a short friendly question (in the detected language) when critical info is missing; null otherwise.
- If "is_airbnb_pricing" is false, set all other fields to null and set "follow_up" to a polite redirect in the detected language.
- Infer "accommodates" from context when reasonable ("studio flat" / "סטודיו" → 2, "family villa" / "וילה משפחתית" → 6).
- Amenity flags default to 0 if not mentioned."""

_EXPLANATION_SYSTEM = """\
You are a friendly, knowledgeable Airbnb pricing advisor for {city}.
{language_instruction}
Write a recommendation in 3 short paragraphs based on the data provided.
- Mention the recommended price and why it fits the market.
- Reference 2-3 of the comparable listings by neighbourhood.
- End with one concrete, actionable tip for the host.
- Do NOT mention machine learning, algorithms, or confidence scores.
- Use {currency} as the currency symbol throughout."""

_FALLBACK_EXPLANATION = {
    "he": (
        "בהתבסס על {n} נכסים דומים ב{city}, "
        "אני ממליץ לתמחר את הנכס שלך ב-{currency}{price:.0f} ללילה "
        "(טווח מחירים לנכסים דומים: {currency}{low:.0f}–{currency}{high:.0f} ללילה)."
    ),
    "en": (
        "Based on {n} comparable listings in {city}, "
        "I recommend pricing your property at {currency}{price:.0f}/night "
        "(comparable range: {currency}{low:.0f}–{currency}{high:.0f}/night). "
        "This places you in the {tier} market segment."
    ),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_history(messages: list) -> str:
    """Return the last 3 conversation turns as plain text for the extraction prompt."""
    if not messages:
        return "(no prior messages)"
    lines = []
    for m in messages[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _call_gemini(
    client: genai.Client,
    system: str,
    user_prompt: str,
    json_mode: bool = False,
) -> Optional[str]:
    config_kwargs = dict(
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        system_instruction=system,
    )
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return None


def _currency(city: str) -> str:
    return "£" if city == "london" else "€"


def _property_summary(features: dict) -> str:
    parts = []
    if features.get("room_type"):
        parts.append(features["room_type"])
    if features.get("property_type"):
        parts.append(features["property_type"])
    if features.get("accommodates"):
        parts.append(f"sleeps {features['accommodates']}")
    if features.get("bedrooms"):
        n = features["bedrooms"]
        parts.append(f"{n} bedroom{'s' if n > 1 else ''}")
    if features.get("neighbourhood_cleansed"):
        parts.append(f"in {features['neighbourhood_cleansed']}")
    return ", ".join(parts) if parts else "Airbnb listing"


def _format_comparables(comparables: list, currency: str) -> str:
    lines = []
    for c in comparables:
        lines.append(
            f"- {c['name']} ({c['neighbourhood']}) — "
            f"{c['room_type']}, sleeps {c['accommodates']}: "
            f"{currency}{c['price']:.0f}/night"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_client() -> genai.Client:
    """Instantiate a Gemini client using GEMINI_API_KEY from settings."""
    return genai.Client(api_key=GEMINI_API_KEY)


def extract_features(
    client: genai.Client,
    city_names: list,
    messages: list,
    user_message: str,
) -> tuple:
    """
    Validate domain and extract structured property features from a conversation turn.

    Returns (extracted_dict, None) on success — extracted_dict is always an
    is_airbnb_pricing=True payload ready for feature accumulation.

    Returns (None, error_reply_str) when:
      - the Gemini API is unreachable (timeout / network error)
      - the response cannot be parsed as JSON
      - the message is not an Airbnb pricing question (domain rejection)

    Handles Hebrew and English input; Gemini replies in the same language as the user.
    Conversation history (last 3 turns) is included for multi-turn context.
    """
    extraction_system = _EXTRACTION_SYSTEM.format(cities=", ".join(city_names))
    extraction_user = (
        f"Conversation so far:\n{_format_history(messages)}\n\n"
        f"User's latest message: {user_message}"
    )
    raw = _call_gemini(client, extraction_system, extraction_user, json_mode=True)

    if raw is None:
        return None, "I'm having trouble reaching the AI service right now. Please try again."

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON despite json_mode: %s", raw[:200])
        return None, "Something went wrong parsing the response. Please rephrase and try again."

    if not extracted.get("is_airbnb_pricing"):
        reply = (
            extracted.get("follow_up")
            or "I can only help with Airbnb pricing. Describe your property and city."
        )
        return None, reply

    return extracted, None


def generate_explanation(
    client: genai.Client,
    city: str,
    features: dict,
    rec,
    user_language: str = "he",
    is_cross_city: bool = False,
    source_cities: Optional[list] = None,
) -> str:
    """
    Generate a natural-language explanation of the ML recommendation.

    user_language: "he" for Hebrew (default, primary language), "en" for English.
    is_cross_city: when True, injects a disclaimer that data comes from other cities.
    Does NOT calculate the recommended price — rec.recommended_price comes
    exclusively from the ML pipeline (engine/recommender.py).
    Falls back to a hardcoded template if Gemini is unavailable.
    """
    currency = _currency(city)
    lang = user_language if user_language in ("he", "en") else "he"
    language_instruction = (
        "Respond in Hebrew (עברית)." if lang == "he" else "Respond in English."
    )
    explanation_system = _EXPLANATION_SYSTEM.format(
        city=city.title(), currency=currency, language_instruction=language_instruction
    )

    if is_cross_city:
        cities_str = ", ".join(c.title() for c in (source_cities or []))
        if lang == "he":
            disclaimer = (
                f"אין לנו נתוני שוק ישירים עבור {city.title()}. "
                f"ההמלצה מבוססת על נכסים דומים מ-{cities_str}. "
                "הביטחון בהמלצה זו נמוך יותר."
            )
        else:
            disclaimer = (
                f"We do not currently have direct data for {city.title()}. "
                f"This recommendation is based on similar properties from {cities_str}. "
                "Confidence is lower."
            )
        explanation_system = explanation_system + f"\n\nIMPORTANT: {disclaimer}"
    explanation_user = (
        f"Property: {_property_summary(features)}\n"
        f"Recommended price: {currency}{rec.recommended_price:.0f}/night\n"
        f"Price range of comparable listings: "
        f"{currency}{rec.price_min:.0f}–{currency}{rec.price_max:.0f}/night\n"
        f"Market segment: {rec.tier_label}\n\n"
        f"Comparable listings:\n{_format_comparables(rec.comparables, currency)}"
    )
    explanation = _call_gemini(client, explanation_system, explanation_user)

    if explanation is None:
        tmpl = _FALLBACK_EXPLANATION.get(lang, _FALLBACK_EXPLANATION["he"])
        explanation = tmpl.format(
            n=len(rec.comparables),
            city=city.title(),
            currency=currency,
            price=rec.recommended_price,
            low=rec.price_min,
            high=rec.price_max,
            tier=rec.tier_label,
        )

    return explanation
