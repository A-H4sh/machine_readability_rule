"""AIClient の JSON 応答パースに関するテスト。

API を叩かず、純粋関数（コードフェンス除去・JSON 解釈）の挙動を検証する。
"""

from __future__ import annotations

import pytest

from harunobu.core.ai_client import AIClient, _parse_json_response, _strip_code_fence


class Test_strip_code_fence:
    """`_strip_code_fence` のテスト。"""

    def test_フェンス無しはそのまま返す(self) -> None:
        assert _strip_code_fence('{"ok": true}') == '{"ok": true}'

    def test_前後の空白を除去する(self) -> None:
        assert _strip_code_fence('  {"ok": true}\n') == '{"ok": true}'

    def test_json付きフェンスを除去する(self) -> None:
        text = '```json\n{"ok": true, "n": 7}\n```'
        assert _strip_code_fence(text) == '{"ok": true, "n": 7}'

    def test_言語指定なしフェンスを除去する(self) -> None:
        text = '```\n{"ok": true}\n```'
        assert _strip_code_fence(text) == '{"ok": true}'

    def test_フェンス周辺の余分な空白を許容する(self) -> None:
        text = '  ```json\n{"a": 1}\n```  '
        assert _strip_code_fence(text) == '{"a": 1}'


class Test_parse_json_response:
    """`_parse_json_response` のテスト。"""

    def test_素のjsonを解釈する(self) -> None:
        assert _parse_json_response('{"ok": true, "n": 7}') == {"ok": True, "n": 7}

    def test_フェンス付きjsonを解釈する(self) -> None:
        # Anthropic claude-4.x が返しがちな形式
        text = '```json\n{"ok": true, "n": 7}\n```'
        assert _parse_json_response(text) == {"ok": True, "n": 7}

    @pytest.mark.parametrize("empty", [None, "", "   ", "\n"])
    def test_空応答はNoneを返す(self, empty: str | None) -> None:
        assert _parse_json_response(empty) is None

    def test_不正なjsonは例外を送出する(self) -> None:
        with pytest.raises(ValueError):
            _parse_json_response("not json at all")


class Test_litellm_completion:
    """`_litellm_completion`（temperature フォールバック）のテスト。"""

    def _patch_litellm(self, monkeypatch: pytest.MonkeyPatch, fake: object) -> None:
        import sys
        import types

        module = types.ModuleType("litellm")
        module.completion = fake  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "litellm", module)

    def test_通常はtemperature付きで呼ぶ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []

        def fake(**kwargs):  # noqa: ANN003, ANN202
            calls.append(kwargs)
            return "resp"

        self._patch_litellm(monkeypatch, fake)
        result = AIClient._litellm_completion("openai/gpt-4o", [{"role": "user", "content": "x"}])

        assert result == "resp"
        assert len(calls) == 1
        assert calls[0]["temperature"] == 0.1

    def test_temperature非対応なら外して再試行する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []

        def fake(**kwargs):  # noqa: ANN003, ANN202
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise ValueError("`temperature` is deprecated for this model.")
            return "resp"

        self._patch_litellm(monkeypatch, fake)
        result = AIClient._litellm_completion("anthropic/claude-opus-4-8", [{"role": "user", "content": "x"}])

        assert result == "resp"
        assert len(calls) == 2
        assert "temperature" not in calls[1]

    def test_temperature以外のエラーは送出する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):  # noqa: ANN003, ANN202
            raise ValueError("authentication failed")

        self._patch_litellm(monkeypatch, fake)
        with pytest.raises(ValueError, match="authentication"):
            AIClient._litellm_completion("openai/gpt-4o", [{"role": "user", "content": "x"}])


class Test_think_block:
    """ローカル推論モデルの ``<think>`` 出力への耐性テスト。"""

    def test_先頭のthinkブロックを除去して解釈する(self) -> None:
        text = '<think>\nまず列を見る…\n</think>\n{"ok": true}'
        assert _parse_json_response(text) == {"ok": True}

    def test_thinkとフェンスの併用を解釈する(self) -> None:
        text = '<think>x</think>\n```json\n{"n": 1}\n```'
        assert _parse_json_response(text) == {"n": 1}

    def test_本文中のthink文字列は除去しない(self) -> None:
        assert _parse_json_response('{"s": "<think>a</think>"}') == {"s": "<think>a</think>"}


_ENDPOINT_ENV_VARS = (
    "HARUNOBU_AI_PROVIDER",
    "HARUNOBU_AI_MODEL",
    "HARUNOBU_AI_API_BASE",
    "HARUNOBU_AI_API_KEY",
    "HARUNOBU_AI_JSON_MODE",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
)


class Test_openai_compatible_endpoint:
    """環境変数で OpenAI 互換エンドポイント（ローカル LLM）を指定するテスト。"""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _ENDPOINT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

    def _capture_litellm(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        import sys
        import types

        calls: list[dict] = []

        def fake(**kwargs):  # noqa: ANN003, ANN202
            calls.append(kwargs)
            message = types.SimpleNamespace(content='{"ok": true}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

        module = types.ModuleType("litellm")
        module.completion = fake  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "litellm", module)
        return calls

    def test_未指定なら従来どおりapi_baseを渡さない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        calls = self._capture_litellm(monkeypatch)

        assert AIClient().generate_json("p", method_name="t") == {"ok": True}
        assert "api_base" not in calls[0]
        assert "api_key" not in calls[0]
        assert calls[0]["model"] == "openai/gpt-5.5"
        assert calls[0]["response_format"] == {"type": "json_object"}

    def test_HARUNOBU_AI_API_BASEでローカルLLMへ送る(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://localhost:11434/v1")
        calls = self._capture_litellm(monkeypatch)

        assert AIClient().generate_json("p", method_name="t") == {"ok": True}
        assert calls[0]["api_base"] == "http://localhost:11434/v1"
        assert calls[0]["model"] == "openai/qwen2.5:7b"
        # キー未設定でもダミーキーで送れる（ローカルサーバは検証しない）
        assert calls[0]["api_key"] == "sk-no-key-required"

    def test_画像付き呼び出しもapi_baseを使う(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://vllm:8000/v1")
        calls = self._capture_litellm(monkeypatch)

        result = AIClient().generate_json_with_image("p", b"\x89PNG", method_name="t")
        assert result == {"ok": True}
        assert calls[0]["api_base"] == "http://vllm:8000/v1"

    def test_スラッシュ入りモデル名をプロバイダーと誤認しない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://vllm:8000/v1")
        calls = self._capture_litellm(monkeypatch)

        AIClient().generate_json("p", method_name="t")
        assert calls[0]["model"] == "openai/Qwen/Qwen2.5-7B-Instruct"

    def test_openai接頭辞付きモデル名は二重にしない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_MODEL", "openai/llama3.1")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://localhost:11434/v1")
        calls = self._capture_litellm(monkeypatch)

        AIClient().generate_json("p", method_name="t")
        assert calls[0]["model"] == "openai/llama3.1"

    def test_APIキーの優先順位(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://x/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        calls = self._capture_litellm(monkeypatch)
        AIClient().generate_json("p", method_name="t")
        assert calls[0]["api_key"] == "sk-openai"

        monkeypatch.setenv("HARUNOBU_AI_API_KEY", "sk-harunobu")
        AIClient().generate_json("p", method_name="t")
        assert calls[1]["api_key"] == "sk-harunobu"

    @pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
    def test_OpenAI標準の環境変数も読む(self, monkeypatch: pytest.MonkeyPatch, var: str) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv(var, "http://lmstudio:1234/v1")
        calls = self._capture_litellm(monkeypatch)

        AIClient().generate_json("p", method_name="t")
        assert calls[0]["api_base"] == "http://lmstudio:1234/v1"

    def test_HARUNOBU_AI_API_BASEがOpenAI標準変数より優先(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://other/v1")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://mine/v1")
        calls = self._capture_litellm(monkeypatch)

        AIClient().generate_json("p", method_name="t")
        assert calls[0]["api_base"] == "http://mine/v1"

    @pytest.mark.parametrize("value", ["0", "false", "off"])
    def test_JSON_MODEを無効化するとresponse_formatを送らない(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("HARUNOBU_AI_PROVIDER", "openai")
        monkeypatch.setenv("HARUNOBU_AI_API_BASE", "http://x/v1")
        monkeypatch.setenv("HARUNOBU_AI_JSON_MODE", value)
        calls = self._capture_litellm(monkeypatch)

        assert AIClient().generate_json("p", method_name="t") == {"ok": True}
        assert "response_format" not in calls[0]
