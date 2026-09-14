import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["mcp/server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # List tools
            tools = await session.list_tools()

            print("\n=== MCP TOOLS ===")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Test employee eligibility
            print("\n=== ELIGIBILITY TEST ===")

            result = await session.call_tool(
                "check_employee_eligibility_mcp",
                {
                    "employee_id": "EMP001"
                },
            )

            print(result)

            # Test trip validation
            print("\n=== TRIP VALIDATION TEST ===")

            result = await session.call_tool(
                "validate_trip_mcp",
                {
                    "employee_id": "EMP001",
                    "trip_type": "airport",
                    "amount": 1500,
                    "time": "11:30 AM",
                },
            )

            print(result)

            # Test reimbursement
            print("\n=== REIMBURSEMENT TEST ===")

            result = await session.call_tool(
                "calculate_reimbursement_mcp",
                {
                    "trip_amount": 2500,
                    "policy_limit": 2000,
                },
            )

            print(result)


if __name__ == "__main__":
    asyncio.run(main())