"""
Offline tests for Hebrew-first NLP behavior in nlp/gemini.py.

These tests patch _call_gemini with realistic mock responses so they run
without a real GEMINI_API_KEY. They verify:
  - prompt construction (correct cities, language rules present)
  - extract_features return values (extracted dict vs error tuple)
  - detected_language field for all four test cases
  - domain rejection path
  - generate_explanation language threading
  - fallback explanation in Hebrew and English

Run:
    py -m pytest tests/test_hebrew_nlp.py -v
"""

import json
import types as builtin_types
from unittest.mock import MagicMock, patch

import pytest

import nlp.gemini as gem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CITY_NAMES = ["amsterdam", "london", "rome"]

def _client():
    return MagicMock()

def _mock_extraction(payload: dict):
    """Patch _call_gemini to return payload as JSON for the extraction call."""
    return patch("nlp.gemini._call_gemini", return_value=json.dumps(payload))

def _mock_explanation(text: str):
    """Patch _call_gemini to return plain text for the explanation call."""
    return patch("nlp.gemini._call_gemini", return_value=text)

def _mock_api_down():
    return patch("nlp.gemini._call_gemini", return_value=None)


# ---------------------------------------------------------------------------
# 1. Full Hebrew message
# ---------------------------------------------------------------------------

class TestHebrewFull:
    PAYLOAD = {
        "is_airbnb_pricing": True,
        "detected_language": "he",
        "city": "amsterdam",
        "features": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "accommodates": 4,
            "bathrooms": None,
            "bedrooms": 2,
            "beds": None,
            "minimum_nights": None,
            "neighbourhood_cleansed": None,
            "amenities_count": None,
            "has_wifi": 0, "has_kitchen": 0, "has_tv": 0,
            "has_ac": 0, "has_pool": 0, "has_parking": 0,
            "has_washer": 0, "has_elevator": 0,
        },
        "expected_price": 180,
        "ready_to_recommend": True,
        "follow_up": None,
    }

    def test_returns_extracted_not_error(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [],
                "יש לי דירת 2 חדרים באמסטרדם שמתאימה ל-4 אורחים. המחיר כרגע הוא 180 דולר ללילה."
            )
        assert err is None
        assert extracted is not None

    def test_detected_language_is_hebrew(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["detected_language"] == "he"

    def test_city_amsterdam(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["city"] == "amsterdam"

    def test_bedrooms_2(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["features"]["bedrooms"] == 2

    def test_accommodates_4(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["features"]["accommodates"] == 4

    def test_expected_price_180(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["expected_price"] == 180

    def test_ready_to_recommend(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["ready_to_recommend"] is True

    def test_room_type_entire_home(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירת 2 חדרים...")
        assert extracted["features"]["room_type"] == "Entire home/apt"


# ---------------------------------------------------------------------------
# 2. Hebrew partial message — asks follow-up in Hebrew
# ---------------------------------------------------------------------------

class TestHebrewPartial:
    PAYLOAD = {
        "is_airbnb_pricing": True,
        "detected_language": "he",
        "city": "amsterdam",
        "features": {
            "room_type": None,
            "property_type": "Apartment",
            "accommodates": None,
            "bathrooms": None, "bedrooms": None, "beds": None,
            "minimum_nights": None, "neighbourhood_cleansed": None,
            "amenities_count": None,
            "has_wifi": 0, "has_kitchen": 0, "has_tv": 0,
            "has_ac": 0, "has_pool": 0, "has_parking": 0,
            "has_washer": 0, "has_elevator": 0,
        },
        "expected_price": None,
        "ready_to_recommend": False,
        "follow_up": "כמה אורחים מתאימה הדירה ואיזה סוג חדר היא?",
    }

    def test_returns_extracted_not_error(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [], "יש לי דירה באמסטרדם"
            )
        assert err is None
        assert extracted is not None

    def test_detected_language_hebrew(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירה באמסטרדם")
        assert extracted["detected_language"] == "he"

    def test_not_ready_to_recommend(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירה באמסטרדם")
        assert extracted["ready_to_recommend"] is False

    def test_follow_up_is_in_hebrew(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירה באמסטרדם")
        follow_up = extracted.get("follow_up", "")
        assert follow_up, "follow_up should not be empty for partial input"
        # Verify the follow_up contains Hebrew characters (Unicode range U+0590–U+05FF)
        assert any("֐" <= ch <= "׿" for ch in follow_up), (
            f"follow_up should be in Hebrew, got: {follow_up!r}"
        )

    def test_no_recommendation_object(self):
        """ready_to_recommend=False means app.py must return only reply+session_id."""
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "יש לי דירה באמסטרדם")
        assert extracted["ready_to_recommend"] is False


# ---------------------------------------------------------------------------
# 3. English message — detected_language "en"
# ---------------------------------------------------------------------------

class TestEnglishFull:
    PAYLOAD = {
        "is_airbnb_pricing": True,
        "detected_language": "en",
        "city": "amsterdam",
        "features": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "accommodates": 4,
            "bathrooms": None,
            "bedrooms": 2,
            "beds": None,
            "minimum_nights": None,
            "neighbourhood_cleansed": None,
            "amenities_count": None,
            "has_wifi": 0, "has_kitchen": 0, "has_tv": 0,
            "has_ac": 0, "has_pool": 0, "has_parking": 0,
            "has_washer": 0, "has_elevator": 0,
        },
        "expected_price": 180,
        "ready_to_recommend": True,
        "follow_up": None,
    }

    def test_detected_language_english(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [],
                "I have a 2-bedroom apartment in Amsterdam for 4 guests. Current price is 180 dollars."
            )
        assert err is None
        assert extracted["detected_language"] == "en"

    def test_city_amsterdam(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "English message")
        assert extracted["city"] == "amsterdam"

    def test_ready_to_recommend(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, _ = gem.extract_features(_client(), CITY_NAMES, [], "English message")
        assert extracted["ready_to_recommend"] is True


# ---------------------------------------------------------------------------
# 4. Mixed / transliterated — should prefer Hebrew
# ---------------------------------------------------------------------------

class TestMixedLanguage:
    PAYLOAD = {
        "is_airbnb_pricing": True,
        "detected_language": "he",   # default to Hebrew when mixed/unclear
        "city": "amsterdam",
        "features": {
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "accommodates": 4,
            "bathrooms": None, "bedrooms": None, "beds": None,
            "minimum_nights": None, "neighbourhood_cleansed": None,
            "amenities_count": None,
            "has_wifi": 0, "has_kitchen": 0, "has_tv": 0,
            "has_ac": 0, "has_pool": 0, "has_parking": 0,
            "has_washer": 0, "has_elevator": 0,
        },
        "expected_price": None,
        "ready_to_recommend": True,
        "follow_up": None,
    }

    def test_mixed_defaults_to_hebrew(self):
        with _mock_extraction(self.PAYLOAD):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [],
                "יש לי apartment באמסטרדם for 4 guests"
            )
        assert err is None
        assert extracted["detected_language"] == "he", (
            "Mixed language should default to Hebrew per language rules"
        )


# ---------------------------------------------------------------------------
# 5. Off-topic / domain rejection
# ---------------------------------------------------------------------------

class TestDomainRejection:
    PAYLOAD_HE = {
        "is_airbnb_pricing": False,
        "detected_language": "he",
        "city": None, "features": None, "expected_price": None,
        "ready_to_recommend": False,
        "follow_up": "אני יכול לעזור רק בתמחור נכסי Airbnb. ספר לי על הנכס שלך.",
    }
    PAYLOAD_EN = {
        "is_airbnb_pricing": False,
        "detected_language": "en",
        "city": None, "features": None, "expected_price": None,
        "ready_to_recommend": False,
        "follow_up": "I can only help with Airbnb pricing. Describe your property and city.",
    }

    def test_hebrew_off_topic_returns_error_tuple(self):
        with _mock_extraction(self.PAYLOAD_HE):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [], "מה מזג האוויר באמסטרדם?"
            )
        assert extracted is None
        assert err is not None

    def test_hebrew_off_topic_error_is_in_hebrew(self):
        with _mock_extraction(self.PAYLOAD_HE):
            _, err = gem.extract_features(
                _client(), CITY_NAMES, [], "מה מזג האוויר באמסטרדם?"
            )
        assert any("֐" <= ch <= "׿" for ch in err), (
            f"Off-topic redirect should be in Hebrew, got: {err!r}"
        )

    def test_english_off_topic_returns_error_tuple(self):
        with _mock_extraction(self.PAYLOAD_EN):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [], "What is the weather in Amsterdam?"
            )
        assert extracted is None
        assert err is not None


# ---------------------------------------------------------------------------
# 6. API failure paths
# ---------------------------------------------------------------------------

class TestAPIFailure:
    def test_api_down_returns_error_tuple(self):
        with _mock_api_down():
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [], "יש לי דירה באמסטרדם"
            )
        assert extracted is None
        assert "trouble" in err.lower() or "שגיאה" in err

    def test_invalid_json_returns_error_tuple(self):
        with patch("nlp.gemini._call_gemini", return_value="not valid json {{{{"):
            extracted, err = gem.extract_features(
                _client(), CITY_NAMES, [], "יש לי דירה באמסטרדם"
            )
        assert extracted is None
        assert err is not None


# ---------------------------------------------------------------------------
# 7. generate_explanation — language threading
# ---------------------------------------------------------------------------

class TestExplanationLanguage:
    """Verify that the correct language_instruction is injected into the prompt."""

    def _fake_rec(self):
        rec = MagicMock()
        rec.recommended_price = 150.0
        rec.price_min = 120.0
        rec.price_max = 180.0
        rec.tier_label = "mid"
        rec.comparables = [
            {"name": "Cozy Flat", "neighbourhood": "Centre",
             "room_type": "Entire home/apt", "accommodates": 4, "price": 145.0}
        ]
        return rec

    def test_hebrew_language_instruction_sent_to_gemini(self):
        captured = {}
        def fake_call(client, system, user_prompt, json_mode=False):
            captured["system"] = system
            return "הסבר בעברית"

        with patch("nlp.gemini._call_gemini", side_effect=fake_call):
            gem.generate_explanation(_client(), "amsterdam", {}, self._fake_rec(), "he")

        assert "עברית" in captured["system"] or "Hebrew" in captured["system"], (
            f"Hebrew instruction missing from system prompt: {captured['system'][:200]}"
        )

    def test_english_language_instruction_sent_to_gemini(self):
        captured = {}
        def fake_call(client, system, user_prompt, json_mode=False):
            captured["system"] = system
            return "English explanation"

        with patch("nlp.gemini._call_gemini", side_effect=fake_call):
            gem.generate_explanation(_client(), "amsterdam", {}, self._fake_rec(), "en")

        assert "English" in captured["system"], (
            f"English instruction missing from system prompt: {captured['system'][:200]}"
        )

    def test_fallback_hebrew_when_api_down(self):
        with _mock_api_down():
            result = gem.generate_explanation(
                _client(), "amsterdam", {}, self._fake_rec(), "he"
            )
        assert any("֐" <= ch <= "׿" for ch in result), (
            f"Hebrew fallback should contain Hebrew characters, got: {result!r}"
        )

    def test_fallback_english_when_api_down(self):
        with _mock_api_down():
            result = gem.generate_explanation(
                _client(), "amsterdam", {}, self._fake_rec(), "en"
            )
        assert "comparable" in result.lower() or "recommend" in result.lower(), (
            f"English fallback should be in English, got: {result!r}"
        )

    def test_unknown_language_defaults_to_hebrew_fallback(self):
        with _mock_api_down():
            result = gem.generate_explanation(
                _client(), "amsterdam", {}, self._fake_rec(), "xx"
            )
        assert any("֐" <= ch <= "׿" for ch in result), (
            "Unknown language code should fall back to Hebrew"
        )


# ---------------------------------------------------------------------------
# 8. Extraction prompt content checks
# ---------------------------------------------------------------------------

class TestExtractionPromptContent:
    """Verify the rendered extraction prompt contains required Hebrew guidance."""

    def _rendered_prompt(self):
        """Return the system prompt exactly as it would be sent to Gemini."""
        return gem._EXTRACTION_SYSTEM.format(cities=", ".join(CITY_NAMES))

    def test_prompt_mentions_hebrew(self):
        prompt = self._rendered_prompt()
        assert "Hebrew" in prompt or "עברית" in prompt or "he" in prompt

    def test_prompt_contains_hebrew_city_aliases(self):
        prompt = self._rendered_prompt()
        assert "אמסטרדם" in prompt
        assert "לונדון" in prompt
        assert "רומא" in prompt

    def test_prompt_contains_detected_language_field(self):
        prompt = self._rendered_prompt()
        assert "detected_language" in prompt

    def test_prompt_hebrew_property_terms_present(self):
        prompt = self._rendered_prompt()
        assert "דירה" in prompt
        assert "אורחים" in prompt

    def test_prompt_cities_formatted_correctly(self):
        prompt = self._rendered_prompt()
        for city in CITY_NAMES:
            assert city in prompt
