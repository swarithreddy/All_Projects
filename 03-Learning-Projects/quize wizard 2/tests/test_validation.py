from quiz_wizard.utils.validation import validate_age, validate_name


def test_validate_name():
    assert validate_name("  ") == "Please enter a name."
    assert validate_name("Ada") is None
    assert validate_name("Ada Lovelace") is None


def test_validate_age():
    age, err = validate_age("19")
    assert age == 19 and err is None
    assert validate_age("0")[1] is not None
    assert validate_age("abc")[1] is not None
    assert validate_age("121")[1] is not None
