
import sys
import os

# Mocking normalize_value as it is in config_analyze_app.py
def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    v = str(value).strip().lower()
    if v in ["1", "true", "yes", "enabled", "enable"]:
        return True
    if v in ["0", "false", "no", "disabled", "disable"]:
        return False
    if v.isdigit():
        return int(v)
    return v

# Mocking evaluate as it is in config_analyze_app.py (the version I just wrote)
def evaluate(actual, expected, operator):
    actual_n = normalize_value(actual)
    expected_n = normalize_value(expected)

    if actual_n is None or expected_n is None:
        return None, "Missing value"

    try:
        # Path normalization if both are strings
        if isinstance(actual_n, str) and isinstance(expected_n, str):
            if "\\" in actual_n or "/" in actual_n:
                actual_n = actual_n.replace("\\", "/").lower().strip()
                expected_n = expected_n.replace("\\", "/").lower().strip()

        # Cast for comparison
        if operator == "==":
            return actual_n == expected_n, None
        if operator == "!=":
            return actual_n != expected_n, None
        
        # New: Handle "include" operator (case-insensitive substring match)
        if operator == "include":
            try:
                a_s = str(actual_n).lower()
                e_s = str(expected_n).lower()
                return e_s in a_s, None
            except (ValueError, TypeError):
                return False, "Error during include comparison"
        
        # For numeric comparisons, try float conversion
        if operator in [">=", "<=", ">", "<"]:
            try:
                a_f = float(str(actual_n))
                e_f = float(str(expected_n))
                if operator == ">=": return a_f >= e_f, None
                if operator == "<=": return a_f <= e_f, None
                if operator == ">": return a_f > e_f, None
                if operator == "<": return a_f < e_f, None
            except (ValueError, TypeError):
                # Fallback for string comparison if float fails
                if operator == ">=": return str(actual_n) >= str(expected_n), None
                if operator == "<=": return str(actual_n) <= str(expected_n), None
                if operator == ">": return str(actual_n) > str(expected_n), None
                if operator == "<": return str(actual_n) < str(expected_n), None
    except Exception as e:
        return None, str(e)

    return None, f"Unsupported operator {operator}"

def run_tests():
    test_cases = [
        # (actual, expected, operator, expected_result)
        ("Success and Failure", "Success", "include", True),
        ("Success", "Success", "include", True),
        ("Failure", "Success", "include", False),
        ("Success and Failure", "success", "include", True), # case-insensitive test
        ("1", "1", "==", True),
        ("0", "1", "==", False),
        ("Enabled", "1", "==", True), # normalization test
        ("Disabled", "0", "==", True), # normalization test
        ("5", "3", ">=", True),
        ("2", "3", ">=", False),
        ("C:\\Windows\\System32", "c:/windows/system32", "==", True), # path normalization test
    ]

    print("Running Compliance Logic Tests...")
    all_passed = True
    for actual, expected, operator, expected_res in test_cases:
        actual_res, err = evaluate(actual, expected, operator)
        if actual_res == expected_res:
            print(f"PASS: actual='{actual}', expected='{expected}', op='{operator}' -> {actual_res}")
        else:
            print(f"FAIL: actual='{actual}', expected='{expected}', op='{operator}' -> {actual_res} (Expected {expected_res}). Error: {err}")
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
