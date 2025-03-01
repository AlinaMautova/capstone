import re

def detect_naming_convention(code_line):
    """
    Detects the naming convention used in a given code line.
    """

    # Hungarian Notation: Variable names prefixed with type indicators (e.g., strCustomerName, prodItem)
    hungarian_pattern = r"\b(?:str|dbl|bool|char|obj|prod|cust|num|arr|ptr|flt|lng|btn|chk|grp|lbl|lst|tbl|txt)[A-Z]\w*"

    # Pascal Case: Class and method names following UpperCamelCase (e.g., Customer, GetEmail, AddOrderItem)
    pascal_pattern = r"\b[A-Z][a-zA-Z0-9]*\b"

    # Acronym Notation: Variable names that are entirely uppercase (e.g., SKU, OTP)
    acronym_pattern = r"\b[A-Z]{2,}\b"

    # Check for Acronym first (since acronyms can be mistakenly classified as Pascal)
    if re.search(acronym_pattern, code_line):
        return "Acronym"

    # Check for Hungarian notation
    if re.search(hungarian_pattern, code_line):
        return "Hungarian Notation"

    # Check for Pascal Case
    if re.search(pascal_pattern, code_line):
        return "Pascal Case"

    return "Unknown"

# Example Usage
code_examples = [
    "private String strCustomerName;",  # Hungarian
    "public Customer(int customerID, String customerName, String email)",  # Pascal
    "public String getEmail() { return strEmail; }",  # Pascal
    "private String strSKU;",  # Acronym
    "private Product prodItem;",  # Hungarian
    "public void AddOrderItem(OrderItem orderItem)",  # Pascal
    "private String strOTP;"  # Acronym
]

# Detect and print results
for code in code_examples:
    print(f"{code} -> {detect_naming_convention(code)}")

# Hungarian Notation: Matches prefixes like str, prod, cust, num, etc. 
# Pascal Case: Matches words that start with an uppercase letter (e.g., CustomerName). 
# Acronym: Matches all-uppercase words of 2 or more letters (e.g., SKU, OTP).