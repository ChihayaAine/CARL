from carl.utils.hashing import treatment_hash


def test_treatment_hash_is_stable_across_key_order():
    a = {"id": "VERIFY", "temperature": 0.0, "turns": 2}
    b = {"turns": 2, "temperature": 0.0, "id": "VERIFY"}
    assert treatment_hash(a) == treatment_hash(b)


def test_treatment_hash_changes_with_value():
    a = {"id": "VERIFY", "turns": 2}
    b = {"id": "VERIFY", "turns": 3}
    assert treatment_hash(a) != treatment_hash(b)
