from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

EMPLOYEE_FILE = DATA_DIR / "employees.csv"


# ============================================================
# COUNTRY POLICY LIMITS
# ============================================================

COUNTRY_LIMITS = {
    "India": {
        "currency": "INR",
        "limit": 2000.0,
    },
    "United States": {
        "currency": "USD",
        "limit": 75.0,
    },
}


# ============================================================
# LOAD EMPLOYEE DATA
# ============================================================

def load_employee_data() -> pd.DataFrame:
    """
    Load employee information from employees.csv.

    Returns:
        pandas DataFrame containing employee records.

    Raises:
        FileNotFoundError:
            If employees.csv does not exist.
        ValueError:
            If required columns are missing.
    """

    if not EMPLOYEE_FILE.exists():
        raise FileNotFoundError(
            f"Employee data file not found: {EMPLOYEE_FILE}"
        )

    df = pd.read_csv(EMPLOYEE_FILE)

    required_columns = {
        "employee_id",
        "country",
        "employee_type",
        "status",
        "manager_approval",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "employees.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Clean string columns
    string_columns = [
        "employee_id",
        "country",
        "employee_type",
        "status",
        "manager_approval",
    ]

    for column in string_columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# GET EMPLOYEE
# ============================================================

def get_employee(employee_id: str) -> Optional[Dict[str, Any]]:
    """
    Find an employee by employee ID.

    Employee ID matching is case-insensitive.

    Example:
        get_employee("EMP001")
    """

    if not employee_id:
        return None

    employee_id = str(employee_id).strip().upper()

    df = load_employee_data()

    matches = df[
        df["employee_id"]
        .str.upper()
        .eq(employee_id)
    ]

    if matches.empty:
        return None

    employee = matches.iloc[0]

    return {
        "employee_id": employee["employee_id"],
        "country": employee["country"],
        "employee_type": employee["employee_type"],
        "status": employee["status"],
        "manager_approval": employee["manager_approval"],
    }


# ============================================================
# CHECK EMPLOYEE ELIGIBILITY
# ============================================================

def check_employee_eligibility(
    employee_id: str,
) -> Dict[str, Any]:
    """
    Check whether an employee is eligible for company travel.

    Possible statuses:

        Eligible
        Approval Required
        Not Eligible
        Unknown Employee
    """

    if not employee_id or not str(employee_id).strip():

        return {
            "found": False,
            "employee_id": employee_id,
            "eligible": False,
            "status": "Unknown Employee",
            "message": "Employee ID is required.",
        }

    employee = get_employee(employee_id)

    if employee is None:

        return {
            "found": False,
            "employee_id": str(employee_id).strip().upper(),
            "eligible": False,
            "status": "Unknown Employee",
            "message": (
                "Employee ID was not found. "
                "Eligibility cannot be assumed."
            ),
        }

    status = employee["status"]

    eligible = status == "Eligible"

    approval_required = (
        status == "Approval Required"
    )

    not_eligible = (
        status == "Not Eligible"
    )

    return {
        "found": True,
        "employee_id": employee["employee_id"],
        "country": employee["country"],
        "employee_type": employee["employee_type"],
        "status": status,
        "manager_approval": employee["manager_approval"],
        "eligible": eligible,
        "approval_required": approval_required,
        "not_eligible": not_eligible,
        "message": _eligibility_message(
            employee
        ),
    }


# ============================================================
# ELIGIBILITY MESSAGE
# ============================================================

def _eligibility_message(
    employee: Dict[str, Any],
) -> str:
    """
    Generate a simple human-readable eligibility message.
    """

    status = employee["status"]

    if status == "Eligible":

        return (
            f"{employee['employee_id']} is eligible "
            f"for company travel."
        )

    if status == "Approval Required":

        return (
            f"{employee['employee_id']} requires "
            f"additional approval for company travel."
        )

    if status == "Not Eligible":

        return (
            f"{employee['employee_id']} is not eligible "
            f"for company travel."
        )

    return (
        f"The travel eligibility status for "
        f"{employee['employee_id']} is {status}."
    )


# ============================================================
# COUNTRY LIMIT
# ============================================================

def get_country_limit(
    country: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the standard travel reimbursement limit
    for a country.

    Supported countries:

        India -> INR 2,000
        United States -> USD 75
    """

    if not country:
        return None

    country_clean = str(country).strip().lower()

    for policy_country, policy in COUNTRY_LIMITS.items():

        if policy_country.lower() == country_clean:

            return {
                "country": policy_country,
                "currency": policy["currency"],
                "limit": policy["limit"],
            }

    return None


# ============================================================
# PARSE TIME
# ============================================================

def parse_time(
    time_value: Optional[str],
) -> Optional[Any]:
    """
    Parse common time formats.

    Supported examples:

        23:30
        22:00
        11:30 PM
        10 PM
        06:00
    """

    if time_value is None:
        return None

    time_text = str(time_value).strip()

    if not time_text:
        return None

    from datetime import datetime

    formats = [
        "%H:%M",
        "%H",
        "%I:%M %p",
        "%I %p",
    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                time_text,
                fmt,
            ).time()

        except ValueError:
            continue

    return None


# ============================================================
# LATE-NIGHT CHECK
# ============================================================

def is_late_night(
    time_value: Optional[str],
) -> bool:
    """
    Determine whether a trip occurs during
    the late-night window.

    Policy:
        10 PM to 6 AM
    """

    parsed_time = parse_time(time_value)

    if parsed_time is None:
        return False

    hour = parsed_time.hour

    return hour >= 22 or hour < 6


# ============================================================
# VALIDATE TRIP
# ============================================================

def validate_trip(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a travel request using:

        1. Employee eligibility
        2. Country
        3. Trip amount
        4. Country spending limit
        5. Late-night policy

    The function does NOT claim that an approval
    actually exists. It only determines whether
    approval is required.
    """

    # --------------------------------------------------------
    # Validate employee ID
    # --------------------------------------------------------

    if not employee_id or not str(employee_id).strip():

        return {
            "valid": False,
            "status": "Invalid Request",
            "employee_id": employee_id,
            "message": "Employee ID is required.",
        }


    # --------------------------------------------------------
    # Validate amount
    # --------------------------------------------------------

    try:
        amount = float(amount)

    except (TypeError, ValueError):

        return {
            "valid": False,
            "status": "Invalid Amount",
            "employee_id": employee_id,
            "message": "Trip amount must be a valid number.",
        }


    if amount < 0:

        return {
            "valid": False,
            "status": "Invalid Amount",
            "employee_id": employee_id,
            "amount": amount,
            "message": "Trip amount cannot be negative.",
        }


    # --------------------------------------------------------
    # Get employee
    # --------------------------------------------------------

    employee = get_employee(employee_id)

    if employee is None:

        return {
            "valid": False,
            "status": "Unknown Employee",
            "employee_id": str(employee_id).strip().upper(),
            "message": (
                "Employee ID was not found. "
                "Travel eligibility cannot be assumed."
            ),
        }


    # --------------------------------------------------------
    # Employee information
    # --------------------------------------------------------

    country = employee["country"]

    employee_status = employee["status"]

    employee_approval = (
        employee["manager_approval"]
    )


    # --------------------------------------------------------
    # Country policy
    # --------------------------------------------------------

    country_policy = get_country_limit(country)

    if country_policy is None:

        return {
            "valid": False,
            "status": "Unsupported Country",
            "employee_id": employee["employee_id"],
            "country": country,
            "message": (
                f"No travel spending limit is configured "
                f"for {country}."
            ),
        }


    policy_limit = country_policy["limit"]

    currency = country_policy["currency"]


    # --------------------------------------------------------
    # Check employee status
    # --------------------------------------------------------

    if employee_status == "Not Eligible":

        return {
            "valid": False,
            "status": "Not Eligible",
            "employee_id": employee["employee_id"],
            "country": country,
            "employee_type": employee["employee_type"],
            "amount": amount,
            "currency": currency,
            "policy_limit": policy_limit,
            "late_night": is_late_night(time),
            "message": (
                f"{employee['employee_id']} is not eligible "
                f"for company travel."
            ),
        }


    # --------------------------------------------------------
    # Check late-night trip
    # --------------------------------------------------------

    late_night = is_late_night(time)


    # --------------------------------------------------------
    # Determine whether approval is required
    # --------------------------------------------------------

    approval_required = False

    approval_reasons = []


    # Employee itself requires approval
    if employee_status == "Approval Required":

        approval_required = True

        approval_reasons.append(
            "employee status requires approval"
        )


    # Amount exceeds country limit
    if amount > policy_limit:

        approval_required = True

        approval_reasons.append(
            f"amount exceeds {currency} {policy_limit:g} limit"
        )


    # --------------------------------------------------------
    # Determine final status
    # --------------------------------------------------------

    if approval_required:

        status = "Approval Required"

        reason_text = "; ".join(
            approval_reasons
        )

        message = (
            f"Trip can be processed, but additional "
            f"approval is required because {reason_text}."
        )

    else:

        status = "Eligible"

        message = (
            f"Trip is within the standard "
            f"{currency} {policy_limit:g} limit."
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "valid": True,
        "status": status,
        "employee_id": employee["employee_id"],
        "country": country,
        "employee_type": employee["employee_type"],
        "employee_status": employee_status,
        "manager_approval": employee_approval,
        "trip_type": trip_type,
        "amount": amount,
        "currency": currency,
        "policy_limit": policy_limit,
        "amount_within_limit": amount <= policy_limit,
        "excess_amount": max(
            amount - policy_limit,
            0.0,
        ),
        "approval_required": approval_required,
        "late_night": late_night,
        "time": time,
        "approval_reasons": approval_reasons,
        "message": message,
    }


# ============================================================
# CALCULATE REIMBURSEMENT
# ============================================================

def calculate_reimbursement(
    trip_amount: float,
    policy_limit: float,
) -> Dict[str, Any]:
    """
    Calculate reimbursable amount.

    Rule:
        reimbursable amount =
        lower of actual trip amount and policy limit.

    Example:

        trip_amount = 1500
        policy_limit = 2000

        reimbursable = 1500

    Example:

        trip_amount = 2500
        policy_limit = 2000

        reimbursable = 2000
        excess = 500
        approval required = True
    """

    # --------------------------------------------------------
    # Validate trip amount
    # --------------------------------------------------------

    try:
        trip_amount = float(trip_amount)

    except (TypeError, ValueError):

        return {
            "success": False,
            "status": "Invalid Amount",
            "message": (
                "Trip amount must be a valid number."
            ),
        }


    # --------------------------------------------------------
    # Validate policy limit
    # --------------------------------------------------------

    try:
        policy_limit = float(policy_limit)

    except (TypeError, ValueError):

        return {
            "success": False,
            "status": "Invalid Policy Limit",
            "message": (
                "Policy limit must be a valid number."
            ),
        }


    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    if trip_amount < 0:

        return {
            "success": False,
            "status": "Invalid Amount",
            "message": (
                "Trip amount cannot be negative."
            ),
        }


    if policy_limit < 0:

        return {
            "success": False,
            "status": "Invalid Policy Limit",
            "message": (
                "Policy limit cannot be negative."
            ),
        }


    # --------------------------------------------------------
    # Calculate reimbursement
    # --------------------------------------------------------

    reimbursable_amount = min(
        trip_amount,
        policy_limit,
    )

    excess_amount = max(
        trip_amount - policy_limit,
        0.0,
    )

    approval_required = (
        trip_amount > policy_limit
    )


    # --------------------------------------------------------
    # Message
    # --------------------------------------------------------

    if approval_required:

        message = (
            f"Standard reimbursable amount is "
            f"{reimbursable_amount:g}. "
            f"The excess amount of "
            f"{excess_amount:g} requires additional "
            f"approval before it can be considered "
            f"reimbursable."
        )

    else:

        message = (
            f"The full trip amount of "
            f"{reimbursable_amount:g} is within "
            f"the policy limit."
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "success": True,
        "status": (
            "Approval Required"
            if approval_required
            else "Within Limit"
        ),
        "trip_amount": trip_amount,
        "policy_limit": policy_limit,
        "reimbursable_amount": reimbursable_amount,
        "excess_amount": excess_amount,
        "approval_required": approval_required,
        "message": message,
    }


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EMPLOYEE DATA")
    print("=" * 60)

    try:

        df = load_employee_data()

        print(df.to_string(index=False))

    except Exception as exc:

        print(f"Error loading employee data: {exc}")


    print("\n" + "=" * 60)
    print("ELIGIBILITY TESTS")
    print("=" * 60)

    employee_ids = [
        "EMP001",
        "EMP002",
        "EMP003",
        "EMP004",
        "EMP005",
        "EMP006",
        "EMP999",
    ]

    for employee_id in employee_ids:

        result = check_employee_eligibility(
            employee_id
        )

        print(
            employee_id,
            "->",
            result,
        )


    print("\n" + "=" * 60)
    print("TRIP VALIDATION TESTS")
    print("=" * 60)

    trip_tests = [

        ("EMP001", "airport", 1500, None),

        ("EMP001", "airport", 2500, None),

        ("EMP003", "airport", 50, None),

        ("EMP003", "airport", 100, None),

        ("EMP001", "airport", 1800, "23:30"),

        ("EMP004", "airport", 50, None),

        ("EMP999", "airport", 1500, None),
    ]

    for test in trip_tests:

        result = validate_trip(
            employee_id=test[0],
            trip_type=test[1],
            amount=test[2],
            time=test[3],
        )

        print(
            f"{test} -> {result}"
        )


    print("\n" + "=" * 60)
    print("REIMBURSEMENT TESTS")
    print("=" * 60)

    reimbursement_tests = [
        (1500, 2000),
        (2000, 2000),
        (2500, 2000),
        (50, 75),
        (100, 75),
    ]

    for trip_amount, limit in reimbursement_tests:

        result = calculate_reimbursement(
            trip_amount,
            limit,
        )

        print(
            f"{trip_amount} / {limit} -> {result}"
        )