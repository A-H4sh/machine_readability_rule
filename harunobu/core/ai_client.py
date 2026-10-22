"""統一AIクライアント — LiteLLM / google.genai 二重バックエンド

LiteLLM がインストールされていれば任意プロバイダー（OpenAI, Anthropic, Gemini 等）
および OpenAI 互換エンドポイント（Ollama / vLLM / LM Studio 等のローカル LLM）を利用し、
未インストール時は google.genai で Gemini のみ動作する。
シングルトンパターン。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """Markdown のコードフェンス（```json … ```）を除去する。

    一部モデル（Anthropic claude-4.x 等）は ``response_format=json_object`` 指定でも
    JSON をコードフェンスで囲んで返すため、`json.loads` の前に取り除く。

    Args:
        text: LLM が返した生テキスト。

    Returns:
        フェンスを除去したテキスト（フェンスが無ければ trim のみ）。
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # 先頭フェンス行（```json 等）を除去
    newline = s.find("\n")
    s = s[newline + 1 :] if newline != -1 else s[3:]
    # 末尾フェンスを除去
    s = s.rstrip()
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def _strip_think_block(text: str) -> str:
    """先頭の ``<think>…</think>`` ブロックを除去する。

    Qwen3 / DeepSeek-R1 等のローカル推論モデルは、OpenAI 互換サーバ経由でも
    思考過程を本文先頭に ``<think>`` タグで出力することがあるため取り除く。

    Args:
        text: LLM が返した生テキスト。

    Returns:
        思考ブロックを除去したテキスト（無ければそのまま）。
    """
    return _THINK_BLOCK_RE.sub("", text, count=1)


def _parse_json_response(text: str | None) -> Any | None:
    """LLM のテキスト応答を JSON として解釈する（コードフェンス耐性あり）。

    Args:
        text: LLM が返した生テキスト。空・None の場合は None を返す。

    Returns:
        解析済みオブジェクト。本文が空の場合は None。
    """
    if not text:
        return None
    cleaned = _strip_code_fence(_strip_think_block(text))
    return json.loads(cleaned) if cleaned else None


def _resolve_ai_config() -> tuple[str, str, str | None]:
    """環境変数から (provider, model, fallback_model) を解決する。"""
    provider = os.environ.get("HARUNOBU_AI_PROVIDER", "gemini")
    default_models: dict[str, str] = {
        "gemini": "gemini-3.5-flash",
        "openai": "gpt-5.5",
        "anthropic": "claude-sonnet-4-6",
    }
    model = os.environ.get("HARUNOBU_AI_MODEL", default_models.get(provider, "gemini-3.5-flash"))
    fallback = os.environ.get("HARUNOBU_AI_FALLBACK_MODEL", "gemini-3.5-flash" if provider == "gemini" else None)
    return provider, model, fallback


# OpenAI 互換のローカルサーバ（Ollama / vLLM / LM Studio / llama.cpp 等）は
# API キーを検証しないが、OpenAI クライアントはキー未設定だとエラーにするため埋める
_NO_KEY_PLACEHOLDER = "sk-no-key-required"


def _env_flag(name: str, default: bool) -> bool:
    """真偽値の環境変数を読む（未設定なら default）。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _resolve_endpoint_config(provider: str) -> tuple[str | None, str | None]:
    """環境変数から (api_base, api_key) を解決する。

    優先順位:
        api_base: ``HARUNOBU_AI_API_BASE`` → （provider=openai のみ）``OPENAI_BASE_URL`` → ``OPENAI_API_BASE``
        api_key:  ``HARUNOBU_AI_API_KEY`` → （provider=openai かつ api_base 指定時）``OPENAI_API_KEY``
                  → それも無ければダミーキー

    api_base も api_key も未指定なら (None, None) を返し、LiteLLM の既定動作に任せる。
    """
    api_base = os.environ.get("HARUNOBU_AI_API_BASE") or None
    if api_base is None and provider == "openai":
        api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or None
    api_key = os.environ.get("HARUNOBU_AI_API_KEY") or None
    if api_key is None and api_base is not None and provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or _NO_KEY_PLACEHOLDER
    return api_base, api_key


_CONNECTION_ERROR_NAMES = frozenset({"ConnectError", "ConnectTimeout", "APIConnectionError"})


def _is_connection_error(exc: BaseException) -> bool:
    """例外連鎖のどこかに「サーバへ接続できなかった」例外があるかを判定する。

    LiteLLM は接続失敗を ``InternalServerError`` 等に包み直すため、
    ``__cause__`` / ``__context__`` をたどって根本の例外を見る。
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ConnectionError) or type(cur).__name__ in _CONNECTION_ERROR_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class AIClient:
    """LiteLLM / google.genai 統一クライアント（シングルトン）。"""

    _instance: AIClient | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> AIClient:
        """シングルトンインスタンスを返す（未生成ならスレッドセーフに生成）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """シングルトンインスタンスを破棄する（テスト・再設定用）。"""
        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        self._available: bool | None = None
        self._backend: str | None = None  # "litellm" または "genai"
        self._genai_client: Any = None
        provider, model, fallback = _resolve_ai_config()
        self._provider = provider
        self._model = model
        self._fallback_model = fallback
        self._api_base, self._api_key = _resolve_endpoint_config(provider)
        self._json_mode = _env_flag("HARUNOBU_AI_JSON_MODE", default=True)

    # ─── 公開 API ─────────────────────────────────────────────

    def is_available(self) -> bool:
        """AI 機能が利用可能かを返す。"""
        if self._available is not None:
            return self._available
        if os.environ.get("HARUNOBU_AI_DISABLED", "").lower() in ("true", "1"):
            self._available = False
            return False
        try:
            self._detect_backend()
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def generate_json(self, prompt: str, *, method_name: str) -> Any | None:
        """テキストプロンプトから JSON レスポンスを取得する。"""
        if not self.is_available():
            return None
        try:
            backend = self._detect_backend()
            if backend == "litellm":
                return self._generate_json_litellm(prompt, method_name)
            return self._generate_json_genai(prompt, method_name)
        except Exception:
            self._log_failure("generate_json", method_name)
            return None

    def generate_json_with_image(
        self,
        prompt: str,
        image_data: bytes,
        *,
        method_name: str,
        mime_type: str = "image/png",
    ) -> Any | None:
        """テキスト + 画像1枚から JSON レスポンスを取得する。"""
        return self.generate_json_with_images(
            prompt,
            [(mime_type, image_data)],
            method_name=method_name,
        )

    def generate_json_with_images(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        *,
        method_name: str,
    ) -> Any | None:
        """テキスト + 複数画像から JSON レスポンスを取得する。"""
        if not self.is_available():
            return None
        try:
            backend = self._detect_backend()
            if backend == "litellm":
                return self._generate_image_litellm(prompt, images, method_name)
            return self._generate_image_genai(prompt, images, method_name)
        except Exception:
            self._log_failure("generate_json_with_images", method_name)
            return None

    def _log_failure(self, api_name: str, method_name: str) -> None:
        """AI 呼び出しの失敗をログに残す（except 節の中から呼ぶこと）。

        接続できなかった場合は、原因の切り分けに要る接続先を 1 行で出し、
        トレースバックは DEBUG レベルに落とす（呼び出しごとに数十行並ぶのを防ぐ）。
        """
        if _is_connection_error(sys.exc_info()[1] or Exception()):
            logger.warning(
                "AI %s failed (%s): cannot connect to %s (model=%s). "
                "Check that the LLM server is running and reachable from this process "
                "(inside Docker, 'localhost' is the container itself).",
                api_name,
                method_name,
                self._api_base or f"the default endpoint of provider '{self._provider}'",
                self._get_litellm_model_id() if self._backend == "litellm" else self._model,
            )
            logger.debug("Connection failure detail", exc_info=True)
            return
        logger.exception("AI %s failed (%s)", api_name, method_name)

    # ─── バックエンド検出 ──────────────────────────────────────

    def _detect_backend(self) -> str:
        if self._backend:
            return self._backend
        # Gemini 以外のプロバイダーは litellm が必須
        if self._provider != "gemini":
            try:
                import litellm  # noqa: F401

                self._backend = "litellm"
                self._log_backend()
                return self._backend
            except ImportError:
                raise ImportError(
                    f"Provider '{self._provider}' requires litellm. Install with: uv sync --extra ai"
                ) from None
        # Gemini: litellm を優先し、無ければ google.genai にフォールバック
        try:
            import litellm  # noqa: F401

            self._backend = "litellm"
        except ImportError:
            try:
                import google.genai  # noqa: F401

                self._backend = "genai"
            except ImportError:
                raise ImportError(
                    "Neither litellm nor google-genai is installed. Install with: uv sync --extra ai"
                ) from None
        if self._backend == "genai" and self._api_base:
            logger.warning(
                "HARUNOBU_AI_API_BASE=%s is ignored: google.genai backend does not support it (install litellm)",
                self._api_base,
            )
        self._log_backend()
        return self._backend

    def _log_backend(self) -> None:
        """選択したバックエンドと接続設定をログに残す（API キーは出さない）。"""
        logger.info(
            "AIClient backend=%s provider=%s model=%s api_base=%s api_key=%s json_mode=%s",
            self._backend,
            self._provider,
            self._get_litellm_model_id() if self._backend == "litellm" else self._model,
            self._api_base or "(default)",
            "set" if self._api_key else "(env/default)",
            self._json_mode,
        )

    def _get_genai_client(self) -> Any:
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client()
        return self._genai_client

    def _get_litellm_model_id(self, model: str | None = None) -> str:
        m = model or self._model
        # OpenAI 互換エンドポイントでは "Qwen/Qwen2.5-7B-Instruct" のような
        # "/" を含むモデル名をそのままサーバへ渡す必要がある（"Qwen" を
        # LiteLLM のプロバイダー名と誤認させない）
        if self._api_base and self._provider == "openai":
            return m if m.startswith("openai/") else f"openai/{m}"
        if "/" in m:
            return m
        prefix_map = {"gemini": "gemini", "openai": "openai", "anthropic": "anthropic"}
        prefix = prefix_map.get(self._provider, self._provider)
        return f"{prefix}/{m}"

    # ─── LiteLLM バックエンド ────────────────────────────────────────

    @staticmethod
    def _litellm_completion(
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        json_mode: bool = True,
    ) -> Any:
        """litellm.completion を JSON モードで呼ぶ。

        ``drop_params=True`` で未対応パラメータを除去するが、litellm のモデルマップに
        未登録の新しいモデル（GPT-5 系・Claude Opus 4.x 等）では temperature が
        除去されずエラーになる。その場合は temperature を外して再試行する。

        Args:
            model_id: litellm 形式の model id。
            messages: チャットメッセージ列。
            api_base: OpenAI 互換エンドポイントの URL（None なら LiteLLM の既定）。
            api_key: API キー（None なら LiteLLM が環境変数から読む）。
            json_mode: False なら ``response_format`` を送らない
                （``json_object`` 非対応のサーバ向け）。

        Returns:
            litellm のレスポンスオブジェクト。
        """
        import litellm

        # 失敗のたびに標準出力へ出る "Give Feedback / Get Help" の案内を抑止する
        litellm.suppress_debug_info = True

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "drop_params": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        try:
            return litellm.completion(temperature=0.1, **kwargs)
        except Exception as e:
            if "temperature" in str(e).lower():
                logger.info("Model %s rejects temperature; retrying without it", model_id)
                return litellm.completion(**kwargs)
            raise

    def _generate_json_litellm(self, prompt: str, method_name: str) -> Any | None:
        from harunobu.core.ai_metrics import track_ai_call

        model_id = self._get_litellm_model_id()
        with track_ai_call(method_name, model_id) as tracker:
            response = self._litellm_completion(
                model_id,
                [{"role": "user", "content": prompt}],
                api_base=self._api_base,
                api_key=self._api_key,
                json_mode=self._json_mode,
            )
            tracker.set_tokens_from_response(response)
            return _parse_json_response(response.choices[0].message.content)

    def _generate_image_litellm(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        method_name: str,
    ) -> Any | None:
        from harunobu.core.ai_metrics import track_ai_call

        content: list[dict[str, Any]] = []
        for mime_type, data in images:
            b64 = base64.b64encode(data).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        model_id = self._get_litellm_model_id()
        with track_ai_call(method_name, model_id) as tracker:
            response = self._litellm_completion(
                model_id,
                [{"role": "user", "content": content}],
                api_base=self._api_base,
                api_key=self._api_key,
                json_mode=self._json_mode,
            )
            tracker.set_tokens_from_response(response)
            return _parse_json_response(response.choices[0].message.content)

    # ─── google.genai バックエンド ───────────────────────────────────

    def _generate_json_genai(self, prompt: str, method_name: str) -> Any | None:
        from google.genai import types

        from harunobu.core.ai_metrics import track_ai_call

        client = self._get_genai_client()
        models_to_try = [self._model]
        if self._fallback_model:
            models_to_try.append(self._fallback_model)

        for model in models_to_try:
            try:
                with track_ai_call(method_name, model) as tracker:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    tracker.set_tokens_from_response(response)
                    return _parse_json_response(response.text)
            except Exception:
                if model == models_to_try[-1]:
                    raise
                logger.warning("Model %s failed, trying fallback", model)

    def _generate_image_genai(
        self,
        prompt: str,
        images: list[tuple[str, bytes]],
        method_name: str,
    ) -> Any | None:
        from google.genai import types

        from harunobu.core.ai_metrics import track_ai_call

        client = self._get_genai_client()
        parts: list[Any] = []
        for mime_type, data in images:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(types.Part.from_text(text=prompt))

        models_to_try = [self._model]
        if self._fallback_model:
            models_to_try.append(self._fallback_model)

        for model in models_to_try:
            try:
                with track_ai_call(method_name, model) as tracker:
                    response = client.models.generate_content(
                        model=model,
                        contents=types.Content(parts=parts),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    tracker.set_tokens_from_response(response)
                    return _parse_json_response(response.text)
            except Exception:
                if model == models_to_try[-1]:
                    raise
                logger.warning("Model %s failed for image, trying fallback", model)
