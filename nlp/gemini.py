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

Cities with full local market data: {cities}
Any other city or country is also accepted — it uses a cross-city estimate instead.

Domain rule:
- Set "is_airbnb_pricing" to true for ANY message that mentions a property, home, apartment,
  room, villa, or any real estate in ANY city or country — regardless of whether that location
  is in the list above.
- Set "is_airbnb_pricing" to false ONLY when the message is entirely unrelated to property
  pricing or Airbnb (e.g. cooking, weather, sports). Never set it to false because of
  an unsupported city or country.

Greeting detection:
- Set "is_greeting" to true ONLY when the message is a pure social greeting or opener with
  NO property-related content whatsoever. Examples: "hi", "hello", "hey", "שלום", "מה קורה",
  "מה שלומך", "היי", "בוקר טוב", "ערב טוב", "good morning", "how are you".
- When "is_greeting" is true: also set "is_airbnb_pricing" to false.
  Set "follow_up" to a warm, natural greeting in the detected language that welcomes the user
  and invites them to freely describe their property — do NOT ask for any specific fields
  such as city, room type, number of guests, bedrooms, etc.
- If the message contains ANY property or pricing content alongside the greeting,
  set "is_greeting" to false and treat it as a normal Airbnb message.

Required JSON structure:
{{
  "is_airbnb_pricing": true | false,
  "is_greeting": true | false,
  "detected_language": "he" | "en",
  "listing_scenario": "new_listing" | "existing_listing" | null,
  "city": string | null,
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
    "has_wifi": 0 | 1 | null,
    "has_kitchen": 0 | 1 | null,
    "has_tv": 0 | 1 | null,
    "has_ac": 0 | 1 | null,
    "has_pool": 0 | 1 | null,
    "has_parking": 0 | 1 | null,
    "has_washer": 0 | 1 | null,
    "has_elevator": 0 | 1 | null,
    "review_scores_rating": number | null,
    "review_scores_location": number | null,
    "review_scores_value": number | null,
    "host_is_superhost": 0 | 1 | null,
    "instant_bookable": 0 | 1 | null
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

Scenario detection:
- Set "listing_scenario" to "new_listing" when the user is creating a new Airbnb listing
  or asking what price to set for a property they have not yet listed.
- Set "listing_scenario" to "existing_listing" when the user already has an active Airbnb
  listing and wants to know whether their current price is correct, too high, or too low.
- Set "listing_scenario" to null when it cannot be determined — ask via "follow_up".

Hebrew understanding:
- Recognise Hebrew property terms:
  דירה / דירת (apartment), חדר (room), בית (house), וילה (villa), סטודיו (studio),
  חדרי שינה (bedrooms), מיטות (beds), אורחים / אנשים / נוסעים (guests / accommodates),
  חדר אמבטיה / שירותים (bathroom), שכונה (neighbourhood), מרפסת (balcony).
- Map Hebrew room types:
  דירה שלמה / כל הדירה → "Entire home/apt"
  חדר פרטי → "Private room"
  חדר משותף → "Shared room"
- Recognise Hebrew city names and transliterations; always output the city in lowercase English.
  Examples: אמסטרדם → amsterdam, לונדון → london, רומא → rome, פריז → paris, ברלין → berlin, תל אביב → tel aviv.
  Extract ANY city the user mentions — not only the supported ones listed in "Available cities".
  Supported cities produce more accurate recommendations; any other city still produces a useful estimate.
- Set "city" to null ONLY when the user has not mentioned any city at all — never set it to null just
  because the city is not in the supported list.
- Infer "accommodates" from Hebrew context:
  "סטודיו" → 2, "זוג" → 2, "משפחה" → 4, "וילה משפחתית" → 6.
- Recognise Hebrew scenario signals:
  "רוצה לפרסם" / "רוצה להכניס נכס" / "דירה חדשה" / "לפרסם ב-Airbnb" → new_listing
  "יש לי דירה כבר" / "אני כבר מפרסם" / "המחיר הנוכחי שלי" / "לבדוק את המחיר" → existing_listing
- Recognise Hebrew existing-listing terms:
  סופרהוסט / כוכב-על / מארח מצטיין → host_is_superhost = 1
  הזמנה מיידית / אינסטנט בוקינג → instant_bookable = 1
  ציון ביקורות / דירוג כללי → review_scores_rating
  ציון מיקום → review_scores_location
  ציון ערך / תמורה לכסף → review_scores_value
  מחיר נוכחי / מחיר קיים / המחיר שאני גובה → expected_price

Readiness rules — set "ready_to_recommend" to true ONLY when the exact conditions below are met.
Never set it to true if "listing_scenario" is null.

  For "new_listing" — ALL of the following must be non-null:
    city, neighbourhood_cleansed, property_type, room_type,
    accommodates, bedrooms, beds, bathrooms, minimum_nights,
    has_wifi, has_kitchen, has_tv, has_ac,
    has_pool, has_parking, has_washer, has_elevator.

  For "existing_listing" — ALL of the above PLUS all of these must be non-null:
    expected_price, review_scores_rating, review_scores_location,
    review_scores_value, host_is_superhost, instant_bookable.

Progressive follow-up rules — ask at most ONE question per turn:
- Ask only for the highest-priority field that is still null, in this order:
    1. listing_scenario (if null — ask whether they are creating a new listing or already have one)
    2. city
    3. room_type and property_type (ask for both together if both are unknown)
    4. accommodates, bedrooms, beds, bathrooms (ask for all unknown ones together)
    5. neighbourhood_cleansed
    6. minimum_nights
    7. Amenity flags — if none are known yet, ask about all eight in a single question;
       if some are known, ask only about the remaining null ones together.
    8. (existing_listing only) expected_price — their current nightly rate
    9. (existing_listing only) review_scores_rating
   10. (existing_listing only) review_scores_location and review_scores_value (ask together)
   11. (existing_listing only) host_is_superhost and instant_bookable (ask together)
- Never ask for a field that already has a non-null value.
- Set "follow_up" to null once "ready_to_recommend" is true.

Extraction rules:
- If "is_airbnb_pricing" is false AND "is_greeting" is false: set all other fields to null
  and set "follow_up" to a single brief sentence stating that you can only assist with Airbnb
  property pricing. Do NOT ask any property questions. Do NOT mention specific fields such as
  city, room type, guests, or bedrooms. The message must end after the refusal — no follow-up
  invitation, no question mark at the end.
- Infer "accommodates" from context when reasonable ("studio flat" / "סטודיו" → 2, "family villa" / "וילה משפחתית" → 6).
- Amenity flags: set to 1 if the amenity is confirmed present, 0 if the user says they do not have it, null if it has not been mentioned yet.
- Set "amenities_count" to the count of amenity flags that equal 1 once all eight are known; otherwise null."""

_EXPLANATION_SYSTEM = """\
You are a professional Airbnb Pricing Advisor.
{language_instruction}
Write exactly 2-3 short sentences explaining the price recommendation.
Base your explanation only on the property type, size, and general market conditions.
Do NOT mention: comparable listings, specific listing names, datasets, algorithms, similar cities,
machine learning, similarity calculations, internal rankings, or any technical details.
Do NOT repeat price numbers in your explanation — they are already shown above.
Do NOT include headings, bullet points, or any formatting — plain prose only."""

_DOMAIN_REJECTION = {
    "he": "אני מתמחה אך ורק בתמחור נכסי Airbnb ולא יכול לעזור בנושא זה.",
    "en": "I specialize only in Airbnb property pricing and cannot help with that.",
}

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


def _format_known_features(known_session: dict) -> str:
    """Summarise all fields already collected in this session.

    Injected into every extraction call so Gemini can generate an accurate
    follow-up question even when the relevant turns have fallen outside the
    6-message conversation window.
    """
    parts = []
    if known_session.get("city"):
        parts.append(f"  city: {known_session['city']}")
    if known_session.get("listing_scenario"):
        parts.append(f"  listing_scenario: {known_session['listing_scenario']}")
    if known_session.get("expected_price") is not None:
        parts.append(f"  expected_price: {known_session['expected_price']}")
    for k, v in (known_session.get("last_features") or {}).items():
        if v is not None:
            parts.append(f"  {k}: {v}")
    return "\n".join(parts) if parts else "  (none yet)"


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


def _currency(city: str) -> str:  # noqa: ARG001  (kept for any future use)
    return "$"


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


def _confidence_label(pct: float, lang: str) -> str:
    if pct < 45:
        return "נמוך" if lang == "he" else "Low"
    if pct < 70:
        return "בינוני" if lang == "he" else "Medium"
    return "גבוה" if lang == "he" else "High"


def _price_action(expected: float, price_min: float, price_max: float, lang: str) -> str:
    if expected > price_max:
        return "הורדת מחיר" if lang == "he" else "Decrease Price"
    if expected < price_min:
        return "העלאת מחיר" if lang == "he" else "Increase Price"
    return "המשך במחיר הנוכחי" if lang == "he" else "Keep Current Price"


def _build_rec_header(rec, lang: str, listing_scenario: str = None, expected_price: float = None) -> str:
    if listing_scenario == "existing_listing" and expected_price is not None:
        action = _price_action(expected_price, rec.price_min, rec.price_max, lang)
        if lang == "he":
            return (
                f"מחיר נוכחי: ${expected_price:.0f} ללילה\n"
                f"המלצה: {action}\n"
                f"טווח מומלץ: ${rec.price_min:.0f} – ${rec.price_max:.0f} ללילה"
            )
        return (
            f"Current Price: ${expected_price:.0f} per night\n"
            f"Recommendation: {action}\n"
            f"Recommended Range: ${rec.price_min:.0f} – ${rec.price_max:.0f} per night"
        )
    if lang == "he":
        return (
            f"מחיר מומלץ: ${rec.recommended_price:.0f} ללילה\n"
            f"טווח מומלץ: ${rec.price_min:.0f} – ${rec.price_max:.0f} ללילה"
        )
    return (
        f"Recommended Price: ${rec.recommended_price:.0f} per night\n"
        f"Recommended Range: ${rec.price_min:.0f} – ${rec.price_max:.0f} per night"
    )


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
    known_session: dict = None,
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
    known_session: the full session dict; injected so Gemini sees all accumulated fields
      regardless of how many turns ago they were provided.
    """
    extraction_system = _EXTRACTION_SYSTEM.format(cities=", ".join(city_names))
    known_str = _format_known_features(known_session) if known_session else "  (none yet)"
    extraction_user = (
        f"Already collected in this session "
        f"(these fields are KNOWN — do not ask for them again):\n"
        f"{known_str}\n\n"
        f"Recent conversation (last few turns):\n{_format_history(messages)}\n\n"
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
        if extracted.get("is_greeting"):
            # Pure greeting — return Gemini's warm welcome (no property questions)
            reply = (
                extracted.get("follow_up")
                or "שלום! אשמח לעזור לך לתמחר נכס Airbnb. ספר לי על הנכס שלך."
            )
        else:
            # True off-topic rejection — use hardcoded message to avoid any property questions
            lang = extracted.get("detected_language", "he")
            reply = _DOMAIN_REJECTION.get(lang, _DOMAIN_REJECTION["he"])
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
    listing_scenario: str = None,
    expected_price: float = None,
) -> str:
    """
    Build a structured price recommendation and generate a short explanation paragraph.

    The structured header (price, range, confidence, action) is assembled in Python so
    it is always correct and never leaks internal details.  Gemini is called only to
    write 2-3 plain explanatory sentences based on property characteristics.

    user_language: "he" | "en" — must come from the persisted session language, not the
      current turn detection, to prevent drift.
    listing_scenario / expected_price: used to build the existing-listing variant of the
      header (current price + increase/decrease/keep action).
    Does NOT expose: comparables, dataset details, algorithm names, or similar cities.
    """
    lang = user_language if user_language in ("he", "en") else "he"
    language_instruction = (
        "Respond in Hebrew (עברית)." if lang == "he" else "Respond in English."
    )

    explanation_system = _EXPLANATION_SYSTEM.format(language_instruction=language_instruction)

    if is_cross_city:
        if lang == "he":
            disclaimer = (
                f"אין נתוני שוק ישירים עבור {city.title()}. "
                "ההערכה מבוססת על נכסים דומים ממספר ערים. "
                "ציין בקצרה שמדובר בהערכה ולא בנתון מדויק עבור עיר זו."
            )
        else:
            disclaimer = (
                f"There is no direct market data for {city.title()}. "
                "The estimate is based on similar properties across multiple markets. "
                "Briefly note this is an estimate, not a precise figure for this city."
            )
        explanation_system += f"\n\nIMPORTANT: {disclaimer}"

    explanation_user = (
        f"Property: {_property_summary(features)}\n"
        f"Market segment: {rec.tier_label}\n"
        + (f"Note: cross-city estimate — no direct data for {city.title()}.\n" if is_cross_city else "")
        + "\nWrite 2-3 sentences explaining the price recommendation "
          "based only on property characteristics and market position."
    )

    header = _build_rec_header(rec, lang, listing_scenario, expected_price)
    explanation_text = _call_gemini(client, explanation_system, explanation_user)

    if explanation_text is None:
        return header

    return f"{header}\n\n{explanation_text}"
