
# Mocking parts of evaluate for testing range logic
def evaluate_range(actual, expected, operator):
    actual_n = str(actual).strip()
    expected_n = str(expected).strip()

    if operator == "range":
        try:
            val = float(actual_n)
            low_str, high_str = expected_n.split("-")
            low = float(low_str)
            high = float(high_str)
            return low <= val <= high, None
        except Exception as e:
            return False, f"Invalid range format or value: {actual_n} vs {expected_n}"
    return None, f"Unsupported operator {operator}"

def run_range_tests():
    test_cases = [
        # (actual, expected, operator, expected_result)
        ("10", "5-14", "range", True),
        ("5", "5-14", "range", True),
        ("14", "5-14", "range", True),
        ("4", "5-14", "range", False),
        ("15", "5-14", "range", False),
        ("7.5", "5-14", "range", True),
    ]

    print("Running Range Compliance Tests...")
    all_passed = True
    for actual, expected, operator, expected_res in test_cases:
        actual_res, err = evaluate_range(actual, expected, operator)
        if actual_res == expected_res:
            print(f"PASS: actual='{actual}', expected='{expected}' -> {actual_res}")
        else:
            print(f"FAIL: actual='{actual}', expected='{expected}' -> {actual_res} (Expected {expected_res}). Error: {err}")
            all_passed = False
    
    if all_passed:
        print("\nAll range tests passed successfully!")
    else:
        print("\nSome range tests failed.")

if __name__ == "__main__":
    run_range_tests()
