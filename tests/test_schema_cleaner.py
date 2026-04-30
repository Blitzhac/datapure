"""Tests for SchemaCleaner — types, dates, bools, emails, phones."""
import pandas as pd
import pytest
from datapure.cleaners.schema import SchemaCleaner


def test_strip_leading_trailing_whitespace():
    df = pd.DataFrame({"name": ["  Alice  ", "Bob ", "  Carol"]})
    result = SchemaCleaner(normalize_strings=True).clean(df)
    assert result["name"].tolist() == ["Alice", "Bob", "Carol"]


def test_lowercase_strings():
    df = pd.DataFrame({"city": ["DELHI", "Mumbai", "CHENNAI"]})
    result = SchemaCleaner(lowercase_strings=True).clean(df)
    assert result["city"].tolist() == ["delhi", "mumbai", "chennai"]


def test_date_parsing_iso_format():
    df = pd.DataFrame({"created": ["2024-01-15", "2024-06-30"]})
    result = SchemaCleaner(date_columns=["created"]).clean(df)
    assert pd.api.types.is_datetime64_any_dtype(result["created"])


def test_date_parsing_mixed_formats():
    df = pd.DataFrame({"date": ["2024-01-15", "15/02/2024"]})
    result = SchemaCleaner(date_columns=["date"]).clean(df)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_bool_coercion_yes_no():
    df = pd.DataFrame({"active": ["yes", "no", "YES", "NO"]})
    result = SchemaCleaner(bool_columns=["active"]).clean(df)
    assert result["active"].tolist() == [True, False, True, False]


def test_bool_coercion_true_false_strings():
    df = pd.DataFrame({"flag": ["True", "False", "true", "false"]})
    result = SchemaCleaner(bool_columns=["flag"]).clean(df)
    assert result["flag"].tolist() == [True, False, True, False]


def test_bool_coercion_0_1():
    df = pd.DataFrame({"flag": ["1", "0", "1"]})
    result = SchemaCleaner(bool_columns=["flag"]).clean(df)
    assert result["flag"].tolist() == [True, False, True]


def test_email_validation_adds_valid_col():
    df = pd.DataFrame({"email": ["user@example.com", "not-an-email", "a@b.co"]})
    result = SchemaCleaner(email_columns=["email"]).clean(df)
    assert "email_valid" in result.columns
    assert result["email_valid"].tolist() == [True, False, True]


def test_phone_validation_adds_valid_col():
    df = pd.DataFrame({"phone": ["+91 98765 43210", "abc", "123-456-7890"]})
    result = SchemaCleaner(phone_columns=["phone"]).clean(df)
    assert "phone_valid" in result.columns
    assert result["phone_valid"].iloc[0] == True   # == not is
    assert result["phone_valid"].iloc[1] == False


def test_auto_coerce_string_numbers():
    df = pd.DataFrame({"age": ["25", "30", "35"]})
    result = SchemaCleaner(auto_coerce=True).clean(df)
    assert pd.api.types.is_numeric_dtype(result["age"])


def test_auto_coerce_does_not_touch_mixed_col():
    df = pd.DataFrame({"col": ["25", "hello", "35"]})
    result = SchemaCleaner(auto_coerce=True).clean(df)
    # Mixed column — less than 90% numeric, should not be coerced
    assert not pd.api.types.is_numeric_dtype(result["col"])


def test_does_not_mutate_input():
    df = pd.DataFrame({"name": ["  Alice  "]})
    SchemaCleaner().clean(df)
    assert df["name"].iloc[0] == "  Alice  "