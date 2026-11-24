#!/usr/bin/env python
"""Seed the Acme runtime state that lives in the cluster DB, NOT in git.

A from-scratch `helm install` brings the platform up, but not the deployment-specific runtime
DATA below (it is data, not chart config). Run this ONCE after the stack is up and the gpu1
model endpoint is reachable. From your workstation:

    kubectl exec -i -n <ns> deploy/rag-api -- /app/.venv/bin/python - < scripts/seed_acme_runtime.py

or from inside the api container:

    /app/.venv/bin/python scripts/seed_acme_runtime.py

It seeds, IDEMPOTENTLY (safe to re-run):
  1. Two in-cluster demo SQL databases in the platform's own Postgres (reachable at postgres:5432):
       - swayatta_4   (CRM: accounts / orders / invoices …)
       - servicedesk  (helpdesk: work orders / statuses …)
     These back the unified-KB structured-evidence feature with sample data. The originals lived on
     192.0.2.13, which a k8s cluster can't route to — so every query hung until they were moved
     in-cluster; this reproduces that move.
  2. The KB's read-only, table-allowlisted SQL connections pointing at those in-cluster DBs.
  3. The org-wide GENERATION default = the self-hosted Sarvam-30B (local GPU) on gpu1 — so answers
     use the fast sovereign model rather than falling through to the platform default.

Targets are overridable via env: SEED_ORG, SEED_SQL_KB, SEED_LLM_URL, SEED_LLM_MODEL, SEED_LLM_NAME.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import psycopg

from rag_common.config import get_settings

ORG = os.environ.get("SEED_ORG", "acme")
SQL_KB = os.environ.get("SEED_SQL_KB", f"{ORG}__hr")
LLM_URL = os.environ.get("SEED_LLM_URL", "http://192.0.2.10:8000/v1")
LLM_MODEL = os.environ.get("SEED_LLM_MODEL", "/model")
LLM_NAME = os.environ.get("SEED_LLM_NAME", "Sarvam-30B (local GPU)")

# Each connection's table allowlist (governed NL->SQL only touches these).
SWAYATTA_TABLES = [
    "accounts",
    "billing_order_items",
    "billing_orders",
    "contacts",
    "invoice_payments",
    "leads",
    "opportunities",
    "orders",
    "quotes",
    "tax_invoices",
]
SERVICEDESK_TABLES = [
    "statusdef",
    "prioritydef",
    "departmentdef",
    "categorydef",
    "aaauser",
    "sdorganization",
    "workorder",
    "workorderstates",
]

SWAYATTA_SQL = """
CREATE TABLE IF NOT EXISTS accounts (id serial primary key, name text, industry text, city text, created_at date);
CREATE TABLE IF NOT EXISTS contacts (id serial primary key, account_id int, name text, email text, phone text, title text);
CREATE TABLE IF NOT EXISTS leads (id serial primary key, name text, company text, status text, source text, created_at date);
CREATE TABLE IF NOT EXISTS opportunities (id serial primary key, account_id int, name text, stage text, amount numeric, close_date date);
CREATE TABLE IF NOT EXISTS quotes (id serial primary key, opportunity_id int, quote_number text, total numeric, status text, created_at date);
CREATE TABLE IF NOT EXISTS orders (id serial primary key, account_id int, order_number text, status text, total numeric, order_date date);
CREATE TABLE IF NOT EXISTS billing_orders (id serial primary key, order_id int, account_id int, amount numeric, status text, billed_at date);
CREATE TABLE IF NOT EXISTS billing_order_items (id serial primary key, billing_order_id int, description text, qty int, unit_price numeric, amount numeric);
CREATE TABLE IF NOT EXISTS tax_invoices (id serial primary key, billing_order_id int, invoice_number text, tax_amount numeric, total numeric, invoice_date date);
CREATE TABLE IF NOT EXISTS invoice_payments (id serial primary key, tax_invoice_id int, amount numeric, method text, paid_at date);
INSERT INTO accounts (name,industry,city,created_at) VALUES ('Acme Software','IT Services','Nashik','2025-01-10'),('Acme Manufacturing','Manufacturing','Pune','2025-02-15'),('BluePeak Retail','Retail','Mumbai','2025-03-01') ON CONFLICT DO NOTHING;
INSERT INTO contacts (account_id,name,email,phone,title) VALUES (1,'Alex Mercer','alex@acme.example','9000000001','CTO'),(2,'Nina Alvarez','anita@acme.example','9000000002','Procurement Head'),(3,'Sam Novak','sameer@bluepeak.example','9000000003','Ops Manager') ON CONFLICT DO NOTHING;
INSERT INTO leads (name,company,status,source,created_at) VALUES ('Vikram Shah','NovaTech','qualified','webinar','2025-04-02'),('Priya Nair','GreenLeaf','new','website','2025-04-10') ON CONFLICT DO NOTHING;
INSERT INTO opportunities (account_id,name,stage,amount,close_date) VALUES (1,'Cloud migration','won',1250000,'2025-05-20'),(2,'ERP rollout','negotiation',890000,'2025-07-01') ON CONFLICT DO NOTHING;
INSERT INTO quotes (opportunity_id,quote_number,total,status,created_at) VALUES (1,'Q-2025-001',1250000,'accepted','2025-05-01'),(2,'Q-2025-002',890000,'sent','2025-06-10') ON CONFLICT DO NOTHING;
INSERT INTO orders (account_id,order_number,status,total,order_date) VALUES (1,'ORD-1001','fulfilled',1250000,'2025-05-25'),(2,'ORD-1002','processing',890000,'2025-07-05') ON CONFLICT DO NOTHING;
INSERT INTO billing_orders (order_id,account_id,amount,status,billed_at) VALUES (1,1,1250000,'paid','2025-05-30'),(2,2,890000,'pending','2025-07-10') ON CONFLICT DO NOTHING;
INSERT INTO billing_order_items (billing_order_id,description,qty,unit_price,amount) VALUES (1,'Cloud VM cluster (annual)',1,1250000,1250000),(2,'ERP licenses',10,89000,890000) ON CONFLICT DO NOTHING;
INSERT INTO tax_invoices (billing_order_id,invoice_number,tax_amount,total,invoice_date) VALUES (1,'INV-2025-001',225000,1475000,'2025-05-30'),(2,'INV-2025-002',160200,1050200,'2025-07-10') ON CONFLICT DO NOTHING;
INSERT INTO invoice_payments (tax_invoice_id,amount,method,paid_at) VALUES (1,1475000,'NEFT','2025-06-02') ON CONFLICT DO NOTHING;
"""

SERVICEDESK_SQL = """
CREATE TABLE IF NOT EXISTS statusdef (statusid serial primary key, statusname text);
CREATE TABLE IF NOT EXISTS prioritydef (priorityid serial primary key, priorityname text);
CREATE TABLE IF NOT EXISTS departmentdef (deptid serial primary key, deptname text);
CREATE TABLE IF NOT EXISTS categorydef (categoryid serial primary key, categoryname text);
CREATE TABLE IF NOT EXISTS aaauser (userid serial primary key, firstname text, email text);
CREATE TABLE IF NOT EXISTS sdorganization (orgid serial primary key, name text);
CREATE TABLE IF NOT EXISTS workorder (workorderid serial primary key, title text, requester text, technician text, statusid int, priorityid int, deptid int, categoryid int, createdtime date, resolution text);
CREATE TABLE IF NOT EXISTS workorderstates (workorderid int, statusid int, changedtime date);
INSERT INTO statusdef (statusname) VALUES ('Open'),('In Progress'),('Resolved'),('Closed') ON CONFLICT DO NOTHING;
INSERT INTO prioritydef (priorityname) VALUES ('Low'),('Medium'),('High'),('Urgent') ON CONFLICT DO NOTHING;
INSERT INTO departmentdef (deptname) VALUES ('IT'),('HR'),('Finance'),('Operations') ON CONFLICT DO NOTHING;
INSERT INTO categorydef (categoryname) VALUES ('Hardware'),('Software'),('Network'),('Access Request') ON CONFLICT DO NOTHING;
INSERT INTO aaauser (firstname,email) VALUES ('Alex','alex@acme.example'),('Blake','blake@acme.example'),('Dana','dana@acme.example') ON CONFLICT DO NOTHING;
INSERT INTO sdorganization (name) VALUES ('Acme Software Solutions') ON CONFLICT DO NOTHING;
INSERT INTO workorder (title,requester,technician,statusid,priorityid,deptid,categoryid,createdtime,resolution) VALUES
 ('Laptop not booting','Alex','Blake',3,3,1,1,'2025-06-01','Replaced RAM module'),
 ('VPN access request','Dana','Blake',4,2,2,4,'2025-06-03','Granted VPN profile'),
 ('Email sync failing','Blake','Dana',2,3,1,2,'2025-06-10',NULL),
 ('Printer offline','Alex','Blake',1,1,4,3,'2025-06-12',NULL) ON CONFLICT DO NOTHING;
"""


def _conninfo(dbname: str) -> str:
    """A plain libpq conninfo (drop the +psycopg driver tag) for ``dbname`` on the platform's PG."""
    dsn = get_settings().postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
    base = dsn.rpartition("/")[0]  # strip the app's own database name
    return f"{base}/{dbname}"


def _app_dsn(dbname: str) -> str:
    """An app-style (postgresql+psycopg://) DSN for ``dbname`` — what the SQL connection stores."""
    base = get_settings().postgres_dsn.rpartition("/")[0]
    return f"{base}/{dbname}"


def create_db_and_schema(dbname: str, schema_sql: str) -> None:
    """CREATE DATABASE if missing (needs autocommit), then apply the idempotent schema + sample rows."""
    with psycopg.connect(_conninfo("postgres"), autocommit=True) as c:
        exists = c.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,)).fetchone()
        if exists:
            print(f"  database {dbname}: exists")
        else:
            c.execute(f'CREATE DATABASE "{dbname}"')
            print(f"  database {dbname}: created")
    with psycopg.connect(_conninfo(dbname)) as c:
        for stmt in (s.strip() for s in schema_sql.split(";")):
            if stmt:
                c.execute(stmt)
        c.commit()
    print(f"  database {dbname}: schema + sample data applied")


async def seed_sql_connections() -> None:
    """Register the read-only, table-allowlisted SQL connections on the KB (deduped by name)."""
    from svc_control_plane import sql_query, wiring  # local import: app wiring

    have = (
        {x.name for x in await sql_query.list_connections(SQL_KB)}
        if hasattr(sql_query, "list_connections")
        else {x.name for x in await wiring.sql_connection_store().list(SQL_KB)}
    )
    for name, db, tables, default in (
        ("swayatta_4", "swayatta_4", SWAYATTA_TABLES, True),
        ("servicedesk", "servicedesk", SERVICEDESK_TABLES, False),
    ):
        if name in have:
            print(f"  SQL connection {name!r} on {SQL_KB}: already registered")
            continue
        await sql_query.add_connection(SQL_KB, name, _app_dsn(db), tables, dialect="postgres", make_default=default)
        print(f"  SQL connection {name!r} on {SQL_KB}: registered -> postgres:5432/{db}")


async def seed_org_default_model() -> None:
    """Ensure the org's GENERATION default is the self-hosted Sarvam-30B (local GPU)."""
    from svc_control_plane import api  # local import: app wiring
    from svc_control_plane.model_connection import StoredModelConnection

    conns = await api._load_org_connections(ORG)

    def is_target(c: StoredModelConnection) -> bool:
        return c.role == "generation" and c.model == LLM_MODEL and c.endpoint == LLM_URL

    if not any(is_target(c) for c in conns):
        conns.append(
            StoredModelConnection(
                tenant_id=ORG,
                conn_id=str(uuid.uuid4()),
                name=LLM_NAME,
                endpoint=LLM_URL,
                model=LLM_MODEL,
                header="",
                key_ciphertext="",
                is_default=True,
                role="generation",
            )
        )
        print(f"  org {ORG}: added generation model {LLM_NAME!r} ({LLM_MODEL} @ {LLM_URL})")
    # exactly one generation default: the target
    conns = [c.model_copy(update={"is_default": is_target(c)}) if c.role == "generation" else c for c in conns]
    await api._save_org_connections(ORG, conns)
    print(f"  org {ORG}: generation default -> {LLM_NAME!r}")


async def main() -> None:
    print(f"== seeding runtime state (org={ORG} sql_kb={SQL_KB} llm={LLM_MODEL}@{LLM_URL}) ==")
    print("1) demo SQL databases:")
    create_db_and_schema("swayatta_4", SWAYATTA_SQL)
    create_db_and_schema("servicedesk", SERVICEDESK_SQL)
    print("2) SQL connections:")
    await seed_sql_connections()
    print("3) org generation default:")
    await seed_org_default_model()
    print("done — re-runnable, no-ops on an already-seeded deployment.")


if __name__ == "__main__":
    asyncio.run(main())
