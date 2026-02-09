# streamlit_app.py
from __future__ import annotations

import io
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Spare Parts Tracker", layout="wide")

DATA_DIR = Path(".data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "seguimiento_piezas.db"

# -----------------------------
# Schema (same idea as your Excel)
# -----------------------------
CASE_FIELDS: List[Tuple[str, str]] = [
    ("case_id", "CASE_ID"),
    ("ticket_cns", "TICKET_CNS"),
    ("cliente", "CLIENTE"),
    ("pais_site", "PAIS_SITE"),
    ("modelo_equipo", "MODELO_EQUIPO"),
    ("sn_equipo", "SN_EQUIPO"),
    ("pieza_descripcion", "PIEZA_DESCRIPCION"),
    ("cantidad", "CANTIDAD"),
    ("rq_no", "RQ_NO"),
    ("rma_no", "RMA_NO"),
    ("order_no", "ORDER_NO"),
    ("fecha_solicitud_hq", "FECHA_SOLICITUD_HQ"),
    ("fecha_ack_hq", "FECHA_ACK_HQ"),
    ("eta_fabricacion", "ETA_FABRICACION"),
    ("fecha_envio_hq", "FECHA_ENVIO_HQ"),
    ("metodo_envio", "METODO_ENVIO"),
    ("tracking_no", "TRACKING_NO"),
    ("fecha_recepcion_cns", "FECHA_RECEPCION_CNS"),
    ("estado", "ESTADO"),
    ("subestado", "SUBESTADO"),
    ("garantia", "GARANTIA"),
    ("responsable", "RESPONSABLE"),
    ("ultima_actualizacion", "ULTIMA_ACTUALIZACION"),
    ("comentarios", "COMENTARIOS"),
]
CASE_COLS = [c for c, _ in CASE_FIELDS]
CASE_HEADERS = [h for _, h in CASE_FIELDS]

# You can keep your internal naming, but the report should include these:
REPORT_STATES = {
    "AWAITING",
    "AWAITING_HQ",
    "FABRICATION",
    "FABRICACION",
    "READY FOR DELIVERY",
    "LISTO_PARA_ENVIO",
    "IN DELIVERY",
    "EN_DELIVERY",
    "EN_TRANSITO",
    "DELIVERY",
}

# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def norm(v: Any) -> str:
    return "" if v is None else str(v).strip()

def compute_case_id(ticket: str, sn: str) -> str:
    ticket = norm(ticket)
    sn = norm(sn)
    if ticket and sn:
        return f"{ticket}-{sn}"
    if ticket:
        return ticket
    if sn:
        return sn
    return ""

def safe_int_str(v: str) -> str:
    v = norm(v)
    if not v:
        return ""
    try:
        return str(int(float(v)))
    except Exception:
        return v  # keep raw

def parse_date_any(s: str) -> Optional[datetime]:
    s = norm(s)
    if not s:
        return None
    # Try common formats first (day-first)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # Fallback: pandas parser
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def days_elapsed_from(start: str) -> Optional[int]:
    dt = parse_date_any(start)
    if not dt:
        return None
    today = datetime.now().date()
    return (today - dt.date()).days

# -----------------------------
# DB Layer
# -----------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # cases table (avoid duplicating case_id)
    non_pk_cols = [c for c in CASE_COLS if c != "case_id"]
    cols_sql = ",\n".join([f"{c} TEXT" for c in non_pk_cols])

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            {cols_sql},
            created_at TEXT,
            updated_at TEXT
        );
        """
    )

    # audit log table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            action TEXT NOT NULL,
            table_name TEXT NOT NULL,
            record_id TEXT NOT NULL,
            field TEXT,
            old_value TEXT,
            new_value TEXT,
            source TEXT,
            batch_id TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_record ON audit_log(record_id);")

    conn.commit()
    conn.close()


def log_action(
    conn: sqlite3.Connection,
    *,
    action: str,
    record_id: str,
    field: str = "*",
    old_value: str = "",
    new_value: str = "",
    source: str = "UI",
    batch_id: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO audit_log(ts, action, table_name, record_id, field, old_value, new_value, source, batch_id)
        VALUES (?, ?, 'cases', ?, ?, ?, ?, ?, ?)
        """,
        (now_iso(), action, record_id, field, old_value, new_value, source, batch_id),
    )

def fetch_cases(search: str = "", status: str = "") -> pd.DataFrame:
    conn = get_conn()
    params: List[Any] = []
    q = "SELECT * FROM cases"
    where = []

    if norm(search):
        s = f"%{search.strip()}%"
        where.append(
            "(case_id LIKE ? OR ticket_cns LIKE ? OR sn_equipo LIKE ? OR cliente LIKE ? OR rq_no LIKE ? OR order_no LIKE ? OR tracking_no LIKE ?)"
        )
        params += [s, s, s, s, s, s, s]

    if norm(status):
        where.append("estado = ?")
        params.append(status.strip())

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " ORDER BY updated_at DESC, created_at DESC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df

def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def upsert_case(data: Dict[str, Any], source: str = "UI", batch_id: str = "") -> Tuple[str, bool, Dict[str, Tuple[str, str]]]:
    # normalize incoming
    clean: Dict[str, str] = {c: norm(data.get(c, "")) for c in CASE_COLS}
    if not clean["case_id"]:
        clean["case_id"] = compute_case_id(clean.get("ticket_cns", ""), clean.get("sn_equipo", ""))
    if not clean["case_id"]:
        raise ValueError("CASE_ID is empty. Provide CASE_ID or TICKET_CNS + SN_EQUIPO.")

    clean["cantidad"] = safe_int_str(clean.get("cantidad", ""))

    cid = clean["case_id"]
    conn = get_conn()
    old = conn.execute("SELECT * FROM cases WHERE case_id = ?", (cid,)).fetchone()
    ts = now_iso()

    if old is None:
        # insert
        clean["created_at"] = ts
        clean["updated_at"] = ts

        cols = CASE_COLS + ["created_at", "updated_at"]
        vals = [clean.get(c, "") for c in CASE_COLS] + [ts, ts]
        placeholders = ",".join(["?"] * len(cols))

        conn.execute(
            f"INSERT INTO cases ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        log_action(conn, action="INSERT", record_id=cid, field="*", old_value="", new_value=str(clean), source=source, batch_id=batch_id)
        conn.commit()
        conn.close()
        return cid, True, {}

    # update
    oldd = dict(old)
    changes: Dict[str, Tuple[str, str]] = {}
    for c in CASE_COLS:
        ov = norm(oldd.get(c, ""))
        nv = norm(clean.get(c, ""))
        if ov != nv:
            changes[c] = (ov, nv)

    if changes:
        set_sql = ", ".join([f"{c} = ?" for c in changes.keys()] + ["updated_at = ?"])
        params = [v[1] for v in changes.values()] + [ts, cid]
        conn.execute(f"UPDATE cases SET {set_sql} WHERE case_id = ?", params)

        for c, (ov, nv) in changes.items():
            log_action(conn, action="UPDATE", record_id=cid, field=c, old_value=ov, new_value=nv, source=source, batch_id=batch_id)

        conn.commit()

    conn.close()
    return cid, False, changes

def delete_case(case_id: str) -> None:
    conn = get_conn()
    old = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if not old:
        conn.close()
        return
    conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
    log_action(conn, action="DELETE", record_id=case_id, field="*", old_value=str(dict(old)), new_value="", source="UI")
    conn.commit()
    conn.close()

def fetch_logs(case_id: str = "", limit: int = 1000) -> pd.DataFrame:
    conn = get_conn()
    if norm(case_id):
        df = pd.read_sql_query(
            "SELECT * FROM audit_log WHERE record_id = ? ORDER BY id DESC LIMIT ?",
            conn,
            params=[case_id.strip(), limit],
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
            conn,
            params=[limit],
        )
    conn.close()
    return df

def fetch_statuses() -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT estado
        FROM cases
        WHERE estado IS NOT NULL AND TRIM(estado) <> ''
        ORDER BY estado
        """
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]




# -----------------------------
# CSV import/merge
# -----------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def normalize_import_df(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either standard headers or internal col names
    cols = list(df.columns)

    header_to_col = {h: c for c, h in CASE_FIELDS}
    out = pd.DataFrame()

    for c, h in CASE_FIELDS:
        if h in cols:
            out[c] = df[h].astype(str).fillna("").map(norm)
        elif c in cols:
            out[c] = df[c].astype(str).fillna("").map(norm)
        else:
            out[c] = ""

    # derive case_id if missing
    def _derive(row):
        cid = norm(row["case_id"])
        if cid:
            return cid
        return compute_case_id(row["ticket_cns"], row["sn_equipo"])

    out["case_id"] = out.apply(_derive, axis=1)
    out = out[out["case_id"].map(lambda x: bool(norm(x)))]
    out["cantidad"] = out["cantidad"].map(safe_int_str)
    return out

def analyze_import(local_df: pd.DataFrame, import_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    # duplicates inside import
    dups = import_df["case_id"][import_df["case_id"].duplicated()].unique().tolist()

    local_map = {}
    if not local_df.empty:
        local_map = {norm(r["case_id"]): r for _, r in local_df.iterrows()}

    items = []
    n_new = n_changed = n_same = 0

    for _, r in import_df.iterrows():
        cid = norm(r["case_id"])
        loc = local_map.get(cid)
        if loc is None:
            n_new += 1
            items.append({
                "TYPE": "NEW",
                "CASE_ID": cid,
                "CHANGED_FIELDS": ", ".join(CASE_COLS),
                "TICKET": norm(r["ticket_cns"]),
                "SN": norm(r["sn_equipo"]),
                "STATUS": norm(r["estado"]),
            })
            continue

        changed = []
        for c in CASE_COLS:
            if norm(loc.get(c, "")) != norm(r.get(c, "")):
                changed.append(c)

        if not changed:
            n_same += 1
            continue

        n_changed += 1
        items.append({
            "TYPE": "CHANGED",
            "CASE_ID": cid,
            "CHANGED_FIELDS": ", ".join(changed),
            "TICKET": norm(r["ticket_cns"]),
            "SN": norm(r["sn_equipo"]),
            "STATUS": norm(r["estado"]),
        })

    summary = {"new": n_new, "changed": n_changed, "same": n_same, "dup": len(dups)}
    return pd.DataFrame(items), summary, dups

def merge_import(import_df: pd.DataFrame, prefer_import: bool) -> Tuple[str, int, int]:
    batch_id = f"import-{uuid.uuid4().hex[:8]}"
    applied_new = applied_changed = 0

    # build local cache
    local_df = fetch_cases()
    local_map = {norm(r["case_id"]): r for _, r in local_df.iterrows()} if not local_df.empty else {}

    for _, r in import_df.iterrows():
        cid = norm(r["case_id"])
        local = local_map.get(cid)

        if local is None:
            upsert_case(r.to_dict(), source="IMPORT", batch_id=batch_id)
            applied_new += 1
            continue

        merged = dict(local)  # existing row dict-like
        for c in CASE_COLS:
            imp = norm(r.get(c, ""))
            loc = norm(local.get(c, ""))
            if imp == loc:
                continue

            conflict = bool(imp and loc and imp != loc)
            if prefer_import:
                merged[c] = imp
            else:
                # prefer local on conflicts, but accept non-empty improvements
                if not imp:
                    merged[c] = loc
                elif conflict:
                    merged[c] = loc
                else:
                    merged[c] = imp

        upsert_case(merged, source="IMPORT", batch_id=batch_id)
        applied_changed += 1

    return batch_id, applied_new, applied_changed

# -----------------------------
# Weekly report
# -----------------------------
def is_report_state(s: str) -> bool:
    s = norm(s).upper()
    return s in {x.upper() for x in REPORT_STATES}

def make_weekly_report_df(cases_df: pd.DataFrame) -> pd.DataFrame:
    if cases_df.empty:
        return pd.DataFrame(columns=[
            "SHIPPED?", "START DATE", "DAYS ELAPSED", "N° RQ", "N° RMA",
            "CLIENT", "MODEL", "PART DESCRIPTION", "SERIAL NUMBER", "STATUS"
        ])

    df = cases_df.copy()

    # Filter only states requiring update
    df = df[df["estado"].astype(str).map(is_report_state)]

    # Compute START DATE priority
    def pick_start(row):
        for k in ["fecha_solicitud_hq", "fecha_ack_hq", "ultima_actualizacion", "created_at"]:
            v = norm(row.get(k, ""))
            if v:
                return v
        return ""

    df["START_DATE"] = df.apply(pick_start, axis=1)
    df["DAYS_ELAPSED"] = df["START_DATE"].map(lambda x: days_elapsed_from(x) if x else None)

    # SHIPPED? heuristic
    def shipped_flag(row):
        est = norm(row.get("estado", "")).upper()
        if "DELIVERY" in est or "TRANSITO" in est or est in {"ENVIADO", "IN DELIVERY"}:
            return "YES"
        if norm(row.get("tracking_no", "")) or norm(row.get("fecha_envio_hq", "")):
            return "YES"
        return "NO"

    out = pd.DataFrame({
        "SHIPPED?": df.apply(shipped_flag, axis=1),
        "START DATE": df["START_DATE"],
        "DAYS ELAPSED": df["DAYS_ELAPSED"].fillna(""),
        "N° RQ": df["rq_no"].fillna(""),
        "N° RMA": df["rma_no"].fillna(""),
        "CLIENT": df["cliente"].fillna(""),
        "MODEL": df["modelo_equipo"].fillna(""),
        "PART DESCRIPTION": df["pieza_descripcion"].fillna(""),
        "SERIAL NUMBER": df["sn_equipo"].fillna(""),
        "STATUS": df["estado"].fillna(""),
    })

    # Sort by days desc
    def _sortkey(x):
        try:
            return int(x)
        except Exception:
            return -1
    out["_sort"] = out["DAYS ELAPSED"].map(_sortkey)
    out = out.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    return out

def report_to_html_table(df: pd.DataFrame) -> str:
    # Minimal "Excel-like" HTML (red header + yellow status)
    css = """
    <style>
      table.report { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }
      table.report th, table.report td { border: 1px solid #000; padding: 6px 8px; vertical-align: top; }
      table.report th { background: #d72626; color: #000; font-weight: 700; text-transform: uppercase; }
      table.report td.status { background: #ffe600; font-weight: 700; text-align: center; }
      table.report td.center { text-align: center; }
    </style>
    """
    headers = df.columns.tolist()
    rows_html = []
    for _, r in df.iterrows():
        tds = []
        for h in headers:
            val = "" if pd.isna(r[h]) else str(r[h])
            cls = ""
            if h.upper() == "STATUS":
                cls = ' class="status"'
            elif h.upper() in {"SHIPPED?", "DAYS ELAPSED"}:
                cls = ' class="center"'
            tds.append(f"<td{cls}>{val}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    html = css + "<table class='report'><thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>"
    html += "<tbody>" + "".join(rows_html) + "</tbody></table>"
    return html

# -----------------------------
# UI
# -----------------------------
init_db()

st.title("📦 Spare Parts Tracker (CNS ↔ HQ)")
st.caption("SQLite local + audit logs per change + CSV import/merge like Git + weekly report export (HTML/CSV).")

page = st.sidebar.radio("Navigation", ["Cases", "Import / Merge", "Logs", "Reports"], index=0)

# ---- CASES ----
if page == "Cases":
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search", placeholder="CASE_ID / Ticket / SN / Client / RQ / Order / Tracking")
    with c2:
    statuses = fetch_statuses()  # ya la tienes definida arriba
    status_choice = st.selectbox(
        "Filter by status",
        options=["(All)"] + statuses,
        index=0,
        key="status_choice",
    )
    status_filter = "" if status_choice == "(All)" else status_choice
    with c3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh"):
            st.rerun()

    df = fetch_cases(search=q, status=status_filter)
    st.subheader("Cases")
    st.dataframe(df, use_container_width=True, height=420)

    st.divider()
    st.subheader("Add / Edit Case")

    # Choose mode
    mode = st.radio("Mode", ["Add", "Edit"], horizontal=True)
    existing_ids = df["case_id"].tolist() if not df.empty else []

    if mode == "Edit":
        case_id = st.selectbox("Select CASE_ID", options=existing_ids if existing_ids else ["(none)"])
        initial = get_case(case_id) if case_id and case_id != "(none)" else {}
    else:
        initial = {}

    with st.form("case_form", clear_on_submit=(mode == "Add")):
        cols = st.columns(3)
        form_data: Dict[str, Any] = {}

        # Lay out fields in columns
        for i, (col, header) in enumerate(CASE_FIELDS):
            target = cols[i % 3]
            with target:
                default = norm(initial.get(col, "")) if initial else ""
                form_data[col] = st.text_input(header, value=default)

        submitted = st.form_submit_button("💾 Save")

    if submitted:
        try:
            # If edit, enforce case_id
            if mode == "Edit" and initial and initial.get("case_id"):
                form_data["case_id"] = initial["case_id"]
            cid, is_new, changes = upsert_case(form_data, source="UI")
            if is_new:
                st.success(f"Inserted: {cid}")
            else:
                if changes:
                    st.success(f"Updated: {cid} | fields: {', '.join(changes.keys())}")
                else:
                    st.info("No changes detected.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Delete Case")
    del_id = st.selectbox("Select CASE_ID to delete", options=[""] + existing_ids)
    if st.button("🗑️ Delete selected", disabled=(not del_id)):
        delete_case(del_id)
        st.success(f"Deleted: {del_id}")
        st.rerun()

    st.divider()
    st.subheader("Export")
    export_df = fetch_cases()
    st.download_button(
        "⬇️ Download cases_export.csv",
        data=df_to_csv_bytes(export_df[CASE_COLS + ["created_at", "updated_at"]] if not export_df.empty else pd.DataFrame()),
        file_name="cases_export.csv",
        mime="text/csv",
    )

# ---- IMPORT / MERGE ----
elif page == "Import / Merge":
    st.subheader("Import / Diff / Merge (like Git)")
    st.caption("Upload a CSV from a teammate. We detect NEW / CHANGED / SAME + DUPLICATES inside the CSV. Then merge with a policy.")

    prefer_import = st.radio("Merge policy", ["Prefer IMPORT", "Prefer LOCAL on conflicts"], horizontal=True) == "Prefer IMPORT"

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            import_raw = pd.read_csv(up)
            import_df = normalize_import_df(import_raw)

            local_df = fetch_cases()
            diff_df, summary, dups = analyze_import(local_df if not local_df.empty else pd.DataFrame(columns=CASE_COLS), import_df)

            st.write(
                f"**Summary:** NEW={summary['new']} | CHANGED={summary['changed']} | SAME={summary['same']} | DUPLICATES_IN_CSV={summary['dup']}"
            )
            if dups:
                st.warning(f"Duplicates in CSV (same CASE_ID repeated): {', '.join(dups[:20])}" + (" ..." if len(dups) > 20 else ""))

            st.dataframe(diff_df, use_container_width=True, height=380)

            can_merge = (summary["new"] + summary["changed"]) > 0
            if st.button("✅ Merge into local DB", disabled=not can_merge):
                batch_id, applied_new, applied_changed = merge_import(import_df, prefer_import=prefer_import)
                st.success(f"Merged. Batch={batch_id} | New={applied_new} | Updated={applied_changed}")
                st.info("Go to Logs to see the full audit trail.")
        except Exception as e:
            st.error(str(e))

# ---- LOGS ----
elif page == "Logs":
    st.subheader("Audit Logs")
    st.caption("Every INSERT/UPDATE/DELETE and IMPORT merge is logged per field with batch_id.")

    c1, c2 = st.columns([2, 1])
    with c1:
        log_case = st.text_input("Filter by CASE_ID (optional)")
    with c2:
        limit = st.number_input("Limit", min_value=100, max_value=10000, value=1000, step=100)

    logs_df = fetch_logs(case_id=log_case, limit=int(limit))
    st.dataframe(logs_df, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download audit_log.csv",
        data=df_to_csv_bytes(logs_df),
        file_name="audit_log.csv",
        mime="text/csv",
    )

# ---- REPORTS ----
else:
    st.subheader("Reports")
    st.caption("Weekly report includes only statuses that require HQ update: awaiting / fabrication / ready for delivery / in delivery / delivery.")

    cases_df = fetch_cases()
    report_df = make_weekly_report_df(cases_df)

    st.write("### Spare Parts Weekly Report (table)")
    st.dataframe(report_df, use_container_width=True, height=520)

    # Downloads
    st.download_button(
        "⬇️ Download weekly_report.csv",
        data=df_to_csv_bytes(report_df),
        file_name="spare_parts_weekly_report.csv",
        mime="text/csv",
    )

    html = report_to_html_table(report_df)
    st.download_button(
        "⬇️ Download weekly_report.html (paste into Outlook)",
        data=html.encode("utf-8"),
        file_name="spare_parts_weekly_report.html",
        mime="text/html",
    )

    with st.expander("Show HTML (copy/paste)"):
        st.code(html, language="html")
