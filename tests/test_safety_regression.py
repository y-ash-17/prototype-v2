from app.safety import assess_safety


def test_hinglish_breathing_emergency():
    assert assess_safety("saans nahi aa rahi hai").level == 1


def test_normal_request_is_not_an_emergency():
    assert assess_safety("book me cardiology").level is None
