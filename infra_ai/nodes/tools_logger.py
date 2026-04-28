# tools_logger.py
class ToolCallLogger:
    def __init__(self):
        self.calls = []

    def log(self, tool_name, kwargs):
        self.calls.append({
            "tool": tool_name,
            "args": kwargs
        })

    def reset(self):
        self.calls = []

    def get_calls(self):
        return self.calls

    def count(self):
        return len(self.calls)


# global instance
# tool_logger = ToolCallLogger()