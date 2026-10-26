"""Fake Anthropic client for tests that exercise code calling core.llm.get_client() --
mirrors Shelf Life's tests/test_chat.py approach of never hitting the real API."""

from types import SimpleNamespace


class FakeUsage(SimpleNamespace):
    input_tokens: int = 100
    output_tokens: int = 50


class FakeParseResponse:
    def __init__(self, parsed_output, input_tokens=100, output_tokens=50):
        self.parsed_output = parsed_output
        self.usage = FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class FakeTextBlock(SimpleNamespace):
    type: str = "text"


class FakeCreateResponse:
    def __init__(self, text, input_tokens=100, output_tokens=50):
        self.content = [FakeTextBlock(type="text", text=text)]
        self.usage = FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class FakeMessages:
    def __init__(self, parse_result=None, create_result=None):
        self._parse_result = parse_result
        self._create_result = create_result

    def parse(self, **kwargs):
        return self._parse_result

    def create(self, **kwargs):
        return self._create_result


class FakeAnthropicClient:
    def __init__(self, parse_result=None, create_result=None):
        self.messages = FakeMessages(parse_result=parse_result, create_result=create_result)
