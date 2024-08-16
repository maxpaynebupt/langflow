# REF: https://langfuse.com/docs/sdk/python/low-level-sdk
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from loguru import logger

from langflow.services.tracing.base import BaseTracer
from langflow.services.tracing.schema import Log
from langflow.services.deps import get_settings_service


settings = get_settings_service()


if TYPE_CHECKING:
    from langflow.graph.vertex.base import Vertex
    from langchain.callbacks.base import BaseCallbackHandler


class LangFuseTracer(BaseTracer):
    def __init__(self, trace_name: str, trace_type: str, project_name: str, trace_id: UUID):
        self.spans: dict = {}
        try:
            self._ready: bool = self.setup_langfuse()
            if not self._ready:
                return

        except Exception as e:
            logger.debug(f"Error setting up LangSmith tracer: {e}")
            self._ready = False

    def setup_langfuse(self) -> bool:
        secret_key = settings.settings.langfuse_secret_key
        public_key = settings.settings.langfuse_public_key
        host = settings.settings.langfuse_host

        if not secret_key or not public_key or not host:
            return False

        status_result = False
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(secret_key=secret_key, public_key=public_key, host=host)
            status_result = True

        except ImportError:
            logger.error("Could not import langfuse. Please install it with `pip install langfuse`.")

        return status_result

    def ready(self) -> bool:
        return self._ready

    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
        vertex: Optional["Vertex"] = None,
    ):
        if not self._ready:
            return
        span = self._langfuse.span(
            trace_id=trace_id,
            name=trace_name,
            metadata=metadata,
            input=inputs,
        )
        self.spans[trace_id] = span

    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: Dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: list[Log | dict] = [],
    ):
        if not self._ready:
            return

        if self.spans.get(trace_id):
            self.spans[trace_id].end(
                output=outputs,
                error=error,
                logs=logs,
            )

    def end(
        self,
        inputs: dict[str, Any],
        outputs: Dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self._ready:
            return
        self._langfuse.flush()

    def get_langchain_callback(self) -> Optional["BaseCallbackHandler"]:
        if self._langfuse is None:
            return None
        return None
