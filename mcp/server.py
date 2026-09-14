from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT EXISTING PROJECT TOOLS
# ============================================================

from src.tools import (
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement,
)


# ============================================================
# MCP 2.x
# ============================================================

from mcp.server.mcpserver import MCPServer


# ============================================================
# CREATE MCP SERVER
# ============================================================

server = MCPServer("Travel Policy Tools")


# ============================================================
# TOOL 1 — EMPLOYEE ELIGIBILITY
# ============================================================

@server.tool()
def check_employee_eligibility_mcp(
    employee_id: str,
) -> dict:
    """
    Check whether an employee is eligible
    for company travel.
    """

    return check_employee_eligibility(employee_id)


# ============================================================
# TOOL 2 — VALIDATE TRIP
# ============================================================

@server.tool()
def validate_trip_mcp(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str | None = None,
) -> dict:
    """
    Validate a travel request against
    employee eligibility and country limits.
    """

    return validate_trip(
        employee_id=employee_id,
        trip_type=trip_type,
        amount=amount,
        time=time,
    )


# ============================================================
# TOOL 3 — CALCULATE REIMBURSEMENT
# ============================================================

@server.tool()
def calculate_reimbursement_mcp(
    trip_amount: float,
    policy_limit: float,
) -> dict:
    """
    Calculate the reimbursable amount,
    excess amount, and approval requirement.
    """

    return calculate_reimbursement(
        trip_amount=trip_amount,
        policy_limit=policy_limit,
    )


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    server.run("stdio")