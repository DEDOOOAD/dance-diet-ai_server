from __future__ import annotations

import asyncio
from concurrent import futures
import importlib
import inspect
import os
from typing import Any, Iterable

import grpc

from ai_server.config import APP_NAME, HOST
from ai_server.proto import dance_video_pb2, dance_video_pb2_grpc


DEFAULT_GRPC_HOST = os.getenv("AI_GRPC_HOST", HOST)
DEFAULT_GRPC_PORT = int(os.getenv("AI_GRPC_PORT", "50052"))


def _load_dance_pipeline_module() -> Any:
    return importlib.import_module("ai_server.services.dance_pipeline")

def _run_maybe_awaitable(result: Any) -> Any:
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result

def _select_callable_arguments(target: Any, payload: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    try:
        parameters = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return [payload["video_bytes"]], {}

    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    kwargs: dict[str, Any] = {}

    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.name in payload:
            kwargs[parameter.name] = payload[parameter.name]

    if kwargs or accepts_kwargs:
        if accepts_kwargs:
            for key, value in payload.items():
                kwargs.setdefault(key, value)
        return [], kwargs

    if not parameters:
        return [], {}

    return [payload["video_bytes"]], {}

def _call_with_payload(target: Any, payload: dict[str, Any]) -> Any:
    args, kwargs = _select_callable_arguments(target, payload)
    return _run_maybe_awaitable(target(*args, **kwargs))

def _resolve_stream_factory(module: Any) -> Any | None:
    candidate_names = (
        "create_stream_processor",
        "create_dance_pipeline",
        "create_pipeline",
        "build_pipeline",
        "DanceStreamPipeline",
        "DancePipeline",
        "StreamProcessor",
        "Pipeline",
    )
    for name in candidate_names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None

def _resolve_chunk_handler(target: Any) -> Any | None:
    if callable(target):
        return target

    candidate_names = (
        "consume_chunk",
        "process_chunk",
        "handle_chunk",
        "process_video_chunk",
        "consume_video_chunk",
        "push_chunk",
    )
    for name in candidate_names:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    return None

def _resolve_finalizer(target: Any) -> Any | None:
    for name in ("finalize", "finish", "close"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    return None

def _create_pipeline_session() -> Any:
    module = _load_dance_pipeline_module()
    factory = _resolve_stream_factory(module)
    if factory is not None:
        return _call_with_payload(factory, {"mode": "stream", "video_bytes": b""})

    if _resolve_chunk_handler(module) is not None:
        return module

    raise RuntimeError("dance_pipeline stream processor is not configured yet.")

def _process_stream_chunk(session: Any, video_bytes: bytes, chunk_index: int, timestamp_ms: int) -> Any:
    chunk_handler = _resolve_chunk_handler(session)
    if chunk_handler is None:
        raise RuntimeError("dance_pipeline chunk handler is not configured yet.")

    payload = {
        "video_bytes": video_bytes,
        "video_chunk": video_bytes,
        "chunk_bytes": video_bytes,
        "chunk": video_bytes,
        "data": video_bytes,
        "payload": video_bytes,
        "chunk_index": chunk_index,
        "timestamp_ms": timestamp_ms,
        "is_final": False,
    }
    return _call_with_payload(chunk_handler, payload)

def _finalize_stream(session: Any, chunk_index: int) -> Any | None:
    finalizer = _resolve_finalizer(session)
    if finalizer is None:
        return None

    payload = {
        "video_bytes": b"",
        "video_chunk": b"",
        "chunk_bytes": b"",
        "chunk": b"",
        "data": b"",
        "payload": b"",
        "chunk_index": chunk_index,
        "timestamp_ms": 0,
        "is_final": True,
    }
    return _call_with_payload(finalizer, payload)

def _extract_result_value(result: Any, field_names: Iterable[str]) -> Any:
    if result is None:
        return None

    if isinstance(result, dict):
        for field_name in field_names:
            value = result.get(field_name)
            if value is not None:
                return value
        return None

    for field_name in field_names:
        if hasattr(result, field_name):
            value = getattr(result, field_name)
            if value is not None:
                return value
    return None

def _build_progress_response(
    result: Any,
    chunk_index: int,
    *,
    fallback_message: str,
    is_final: bool,
) -> dance_video_pb2.DanceVideoProgress:
    success = _extract_result_value(result, ("success", "ok", "accepted"))
    message = _extract_result_value(result, ("message", "detail", "status"))
    calories_burned = _extract_result_value(
        result,
        ("calories_burned", "current_calories", "calories", "burned_calories"),
    )
    total_calories = _extract_result_value(
        result,
        ("total_calories", "accumulated_calories", "cumulative_calories", "sum_calories"),
    )
    movement_score = _extract_result_value(result, ("movement_score", "score"))
    current_met = _extract_result_value(result, ("current_met", "met"))

    if isinstance(result, str) and result.strip():
        message = result.strip()
    elif isinstance(result, (int, float)):
        calories_burned = float(result)
        total_calories = float(result) if total_calories is None else total_calories

    return dance_video_pb2.DanceVideoProgress(
        success=True if success is None else bool(success),
        message=message or fallback_message,
        chunk_index=chunk_index,
        is_final=is_final,
        calories_burned=0.0 if calories_burned is None else float(calories_burned),
        total_calories=0.0 if total_calories is None else float(total_calories),
        movement_score=0.0 if movement_score is None else float(movement_score),
        current_met=0.0 if current_met is None else float(current_met),
    )

class DanceVideoStreamServicer(dance_video_pb2_grpc.DanceVideoStreamServiceServicer):
    def StreamDanceVideo(
        self,
        request_iterator: Iterable[dance_video_pb2.DanceVideoChunk],
        context: grpc.ServicerContext,
    ) -> Iterable[dance_video_pb2.DanceVideoProgress]:
        try:
            session = _create_pipeline_session()
        except Exception as exc:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"dance_pipeline is not ready: {exc}",
            )

        chunk_index = 0
        received_any = False

        for request in request_iterator:
            if not request.video_chunk:
                continue

            received_any = True
            resolved_chunk_index = request.chunk_index if request.chunk_index > 0 else chunk_index + 1
            chunk_index = resolved_chunk_index

            try:
                result = _process_stream_chunk(
                    session,
                    video_bytes=bytes(request.video_chunk),
                    chunk_index=resolved_chunk_index,
                    timestamp_ms=request.timestamp_ms,
                )
            except Exception as exc:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"dance_pipeline chunk failed at chunk {resolved_chunk_index}: {exc}",
                )

            yield _build_progress_response(
                result,
                resolved_chunk_index,
                fallback_message="chunk processed",
                is_final=False,
            )

        if not received_any:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Video request stream was empty.")

        final_result = _finalize_stream(session, chunk_index)
        if final_result is not None:
            yield _build_progress_response(
                final_result,
                chunk_index,
                fallback_message="stream completed",
                is_final=True,
            )

def serve(host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dance_video_pb2_grpc.add_DanceVideoStreamServiceServicer_to_server(DanceVideoStreamServicer(), server)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to {host}:{port}.")

    server.start()
    print(f"{APP_NAME} gRPC server listening on {host}:{bound_port}")
    print("registered: ai_server.proto.DanceVideoStreamService/StreamDanceVideo")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
