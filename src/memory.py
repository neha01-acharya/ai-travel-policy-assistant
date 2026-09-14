from __future__ import annotations

from typing import Any


class ConversationMemory:

    def __init__(self):
        self.history: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(
        self,
        role: str,
        message: str,
    ):

        self.history.append(
            {
                "role": role,
                "message": message,
            }
        )

    # ========================================================
    # UPDATE CONTEXT
    # ========================================================

    def update_context(
        self,
        **kwargs,
    ):

        for key, value in kwargs.items():

            if value is not None:
                self.context[key] = value

    # ========================================================
    # GET CONTEXT
    # ========================================================

    def get_context(self) -> dict[str, Any]:

        return self.context.copy()

    # ========================================================
    # GET HISTORY
    # ========================================================

    def get_history(self) -> list[dict[str, Any]]:

        return self.history.copy()

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):

        self.history.clear()
        self.context.clear()

    # ========================================================
    # BUILD CONTEXT STRING
    # ========================================================

    def context_for_prompt(self) -> str:

        if not self.context:
            return "No previous travel context."

        lines = []

        for key, value in self.context.items():

            lines.append(
                f"{key}: {value}"
            )

        return "\n".join(lines)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    memory = ConversationMemory()

    memory.add_message(
        "user",
        "Can EMP001 take an airport trip costing ₹1,500?",
    )

    memory.add_message(
        "assistant",
        "Yes, the trip is within the standard limit.",
    )

    memory.update_context(
        employee_id="EMP001",
        trip_type="airport",
        amount=1500,
        country="India",
    )

    print("\nCONVERSATION HISTORY:")
    print(memory.get_history())

    print("\nCURRENT CONTEXT:")
    print(memory.get_context())

    print("\nPROMPT CONTEXT:")
    print(memory.context_for_prompt())

    print("\nUpdating amount...")

    memory.update_context(
        amount=2500,
    )

    print(memory.get_context())