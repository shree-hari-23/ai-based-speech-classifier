"""
History Service for Speech Emotion Recognition (EmotiSound AI).
Persists and manages prediction records in Fluxbase Cloud Database.
Supports searching, filtering by emotion/model, pagination, and deletion.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from services.fluxbase_client import (
    execute_sql,
    sql_quote,
    init_fluxbase_schema,
    is_fluxbase_configured
)

logger = logging.getLogger("SER_HistoryService")


def init_db():
    """Initializes the prediction_history schema in Fluxbase."""
    init_fluxbase_schema()


def save_prediction(
    filename: str,
    predicted_emotion: str,
    confidence: float,
    model_used: str,
    probabilities: Dict[str, float],
    file_size: Optional[int] = None,
    duration: Optional[float] = None,
    waveform_sample: Optional[List[float]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Saves a new speech prediction record to the Fluxbase Cloud Database.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    probs_json = json.dumps(probabilities)
    waveform_json = json.dumps(waveform_sample) if waveform_sample else None

    # Safe SQL insertion without PostgreSQL-specific RETURNING clause (compatible with Fluxbase)
    insert_sql = f"""
    INSERT INTO prediction_history (
        filename, file_size, duration, predicted_emotion,
        confidence, model_used, probabilities, waveform_sample, created_at, user_id
    ) VALUES (
        {sql_quote(filename)},
        {sql_quote(file_size)},
        {sql_quote(duration)},
        {sql_quote(predicted_emotion)},
        {sql_quote(round(confidence, 2))},
        {sql_quote(model_used)},
        {sql_quote(probs_json)},
        {sql_quote(waveform_json)},
        {sql_quote(created_at)},
        {sql_quote(user_id)}
    );
    """

    record_id = int(datetime.now().timestamp() * 1000) % 1000000
    try:
        execute_sql(insert_sql)
    except Exception as e:
        logger.warning(f"Could not persist prediction to Fluxbase: {e}")

    return {
        "id": record_id or 1,
        "filename": filename,
        "file_size": file_size,
        "duration": duration,
        "predicted_emotion": predicted_emotion,
        "confidence": round(confidence, 2),
        "model_used": model_used,
        "probabilities": probabilities,
        "waveform_sample": waveform_sample,
        "created_at": created_at,
        "user_id": user_id,
        "storage": "Fluxbase Cloud Database" if is_fluxbase_configured() else "Local Cache"
    }


def get_history(
    search: Optional[str] = None,
    emotion: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Retrieves prediction history from Fluxbase with optional search query,
    emotion filter, model filter, and pagination.
    """
    where_clauses = ["1=1"]

    if search:
        s_clean = search.replace("'", "''")
        where_clauses.append(f"(filename ILIKE '%{s_clean}%' OR predicted_emotion ILIKE '%{s_clean}%')")

    if emotion and emotion != "all":
        e_clean = emotion.replace("'", "''")
        where_clauses.append(f"LOWER(predicted_emotion) = LOWER('{e_clean}')")

    if model and model != "all":
        m_clean = model.replace("'", "''")
        where_clauses.append(f"LOWER(model_used) = LOWER('{m_clean}')")

    where_str = " AND ".join(where_clauses)

    # 1. Count query
    count_sql = f"SELECT COUNT(*) AS total FROM prediction_history WHERE {where_str};"
    total_count = 0
    try:
        count_rows = execute_sql(count_sql)
        if count_rows and len(count_rows) > 0:
            total_count = int(count_rows[0].get("total", count_rows[0].get("count", 0)))
    except Exception as e:
        logger.warning(f"Count query failed on Fluxbase: {e}")

    # 2. Rows query
    query_sql = f"""
    SELECT id, filename, file_size, duration, predicted_emotion,
           confidence, model_used, probabilities, waveform_sample, created_at, user_id
    FROM prediction_history
    WHERE {where_str}
    ORDER BY id DESC
    LIMIT {limit} OFFSET {offset};
    """

    rows = []
    try:
        rows = execute_sql(query_sql)
        if total_count == 0 and len(rows) > 0:
            total_count = len(rows)
    except Exception as e:
        logger.error(f"Failed to query prediction_history on Fluxbase: {e}")
        rows = []

    items = []
    for r in rows:
        probs = r.get("probabilities")
        if isinstance(probs, str):
            try:
                probs = json.loads(probs)
            except Exception:
                probs = {}
        elif not isinstance(probs, dict):
            probs = {}

        wf = r.get("waveform_sample")
        if isinstance(wf, str):
            try:
                wf = json.loads(wf)
            except Exception:
                wf = []
        elif not isinstance(wf, list):
            wf = []

        items.append({
            "id": r.get("id"),
            "filename": r.get("filename"),
            "file_size": r.get("file_size"),
            "duration": float(r["duration"]) if r.get("duration") is not None else None,
            "predicted_emotion": r.get("predicted_emotion"),
            "confidence": float(r["confidence"]) if r.get("confidence") is not None else 0.0,
            "model_used": r.get("model_used"),
            "probabilities": probs,
            "waveform_sample": wf,
            "created_at": r.get("created_at"),
            "user_id": r.get("user_id"),
        })

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "items": items,
        "provider": "Fluxbase Cloud Database"
    }


def delete_history_item(record_id: int) -> bool:
    """Deletes a specific history record from Fluxbase by ID."""
    sql = f"DELETE FROM prediction_history WHERE id = {int(record_id)};"
    try:
        execute_sql(sql)
        return True
    except Exception as e:
        logger.error(f"Failed to delete prediction record #{record_id} from Fluxbase: {e}")
        return False


def clear_all_history() -> int:
    """Deletes all records from prediction_history in Fluxbase."""
    sql = "DELETE FROM prediction_history;"
    try:
        execute_sql(sql)
        return 1
    except Exception as e:
        logger.error(f"Failed to clear prediction history from Fluxbase: {e}")
        return 0
