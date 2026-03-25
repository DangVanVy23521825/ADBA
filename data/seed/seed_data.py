"""
ADBA — Seed Data Generator
==========================
Generates realistic, causally-consistent synthetic data for 3 domains:
  Sales    : customers, products, orders
  Inventory: warehouses, stock, stock_movements
  HR       : departments, employees, payroll

Key design principles:
  1. Causality  — events have traceable causes across domains
  2. Consistency — region weights, seasonal multipliers, and product IDs
                   are shared across all three domains
  3. Anomalies  — 15 named anomalies are injected deliberately;
                  each is documented in ANOMALY_CATALOGUE at the bottom
  4. Constraints — respects every CHECK, UNIQUE, GENERATED ALWAYS, and
                   DEFERRABLE FK defined in the schema files
"""

from __future__ import annotations

import os
import random
import math
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
from faker import Faker
from dotenv import load_dotenv

load_dotenv()

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("vi_VN")
fake.seed_instance(SEED)

# ── connection ────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or (
    "postgresql://adba_user:adba_password@localhost:5432/adba_db"
)

# =============================================================================
# SHARED CONSTANTS — used by ALL three domains to guarantee consistency
# =============================================================================

REGIONS = ["Miền Bắc", "Miền Trung", "Miền Nam", "Tây Nguyên"]

# Sales orders, warehouse capacity, and HR headcount all follow these weights
REGION_WEIGHTS = {
    "Miền Bắc":   0.40,
    "Miền Trung": 0.20,
    "Miền Nam":   0.30,
    "Tây Nguyên": 0.10,
}

# HR headcount is slightly different — Tây Nguyên is over-staffed vs revenue
# (intentional: creates "Region X is over-staffed" insight)
REGION_HEADCOUNT_WEIGHTS = {
    "Miền Bắc":   0.35,
    "Miền Trung": 0.20,
    "Miền Nam":   0.30,
    "Tây Nguyên": 0.15,   # 15% headcount but only 10% revenue → insight H-X1
}

DATE_START = date(2022, 1, 1)
DATE_END   = date(2024, 12, 31)

# Seasonal multipliers apply to orders, stock_movements, and payroll overtime
SEASONAL: dict[int, dict[str, float]] = {
    1: {"sales": 0.78, "movements": 0.80, "overtime": 0.85},
    2: {"sales": 0.82, "movements": 0.83, "overtime": 0.87},
    3: {"sales": 0.95, "movements": 0.92, "overtime": 0.95},
    4: {"sales": 1.00, "movements": 1.00, "overtime": 1.00},
    5: {"sales": 1.05, "movements": 1.02, "overtime": 1.05},
    6: {"sales": 1.08, "movements": 1.05, "overtime": 1.08},
    7: {"sales": 1.10, "movements": 1.06, "overtime": 1.10},
    8: {"sales": 1.12, "movements": 1.08, "overtime": 1.12},
    9: {"sales": 1.15, "movements": 1.10, "overtime": 1.12},
    10: {"sales": 1.20, "movements": 1.15, "overtime": 1.15},
    11: {"sales": 1.28, "movements": 1.22, "overtime": 1.20},
    12: {"sales": 1.35, "movements": 1.30, "overtime": 1.25},
}

# =============================================================================
# ANOMALY CATALOGUE
# Referenced by seed functions — do NOT change IDs after initial seed
# =============================================================================

ANOMALY_CATALOGUE = {
    # ── Sales ─────────────────────────────────────────────────────────────────
    "S1": {
        "name": "Revenue spike March 2024 (Miền Bắc)",
        "domain": "sales",
        "type": "positive_outlier",
        "description": (
            "Mega B2B campaign drives +280% revenue in Miền Bắc during March 2024. "
            "Only Enterprise segment. Correlated with I2 (stockout) and H4 (overtime)."
        ),
        "multiplier": 2.80,
        "filter": lambda r, m, y, seg: (
            r == "Miền Bắc" and y == 2024 and m == 3 and seg == "Enterprise"
        ),
    },
    "S2": {
        "name": "Negative-margin product ELEC-007",
        "domain": "sales",
        "type": "negative",
        "description": (
            "SKU ELEC-007 sold with discount_rate so high that effective price < cost "
            "from Q2 2024 onward. Represents a pricing mistake."
        ),
        "sku": "ELEC-007",
        "start_date": date(2024, 4, 1),
        "discount_rate": Decimal("0.4500"),   
    },
    "S3": {
        "name": "Enterprise churn Q3 2024",
        "domain": "sales",
        "type": "negative",
        "description": (
            "20 Enterprise customers place no orders after July 1 2024. "
            "They contributed ~15% of 2023 revenue."
        ),
        "churned_customer_count": 20,
        "churn_date": date(2024, 7, 1),
    },
    "S4": {
        "name": "Refund surge November 2023 (Electronics)",
        "domain": "sales",
        "type": "negative",
        "description": (
            "Refund rate for Electronics category hits 3.4× baseline in Nov 2023. "
            "Simulates a defective batch."
        ),
        "multiplier": 3.40,
        "category": "Electronics",
        "month": 11,
        "year": 2023,
    },
    "S5": {
        "name": "Payment method shift to e-wallet Q2 2024",
        "domain": "sales",
        "type": "structural",
        "description": (
            "e_wallet share grows from ~5% to ~35% of orders from Q2 2024 onward."
        ),
        "start_date": date(2024, 4, 1),
        "target_share": 0.35,
    },
    # ── Inventory ─────────────────────────────────────────────────────────────
    "I1": {
        "name": "Stockout risk — 6 SKUs below min_threshold",
        "domain": "inventory",
        "type": "negative",
        "description": "6 products have quantity < min_threshold at ≥1 warehouse.",
        "count": 6,
    },
    "I2": {
        "name": "Dead stock — 8 SKUs not sold in 180 days",
        "domain": "inventory",
        "type": "warning",
        "description": "8 products show inbound but no outbound in last 180 days.",
        "count": 8,
    },
    "I3": {
        "name": "Write-off spike Q4 2023 (Tây Nguyên)",
        "domain": "inventory",
        "type": "negative",
        "description": "write_off movements 5× baseline at Tây Nguyên warehouse Q4 2023.",
        "multiplier": 5.0,
    },
    "I4": {
        "name": "Inventory-Sales mismatch (Miền Bắc Q4 2024)",
        "domain": "inventory",
        "type": "structural",
        "description": (
            "Outbound movements increase in Miền Bắc Q4 2024 but stock snapshot "
            "does not decrease proportionally — missing inbound records."
        ),
    },
    "I5": {
        "name": "Transfer loop Bắc ↔ Trung",
        "domain": "inventory",
        "type": "warning",
        "description": (
            "Same SKUs transferred Bắc→Trung then back Trung→Bắc within 30 days."
        ),
    },
    # ── HR ────────────────────────────────────────────────────────────────────
    "H1": {
        "name": "Bonus outliers — 4 employees",
        "domain": "hr",
        "type": "structural",
        "description": (
            "4 employees receive bonus > 3σ vs. same-level peers. "
            "2 justified by high performance_score, 2 unexplained."
        ),
        "count": 4,
    },
    "H2": {
        "name": "Attrition cluster Q2 2024 (Tech dept)",
        "domain": "hr",
        "type": "negative",
        "description": (
            "15 employees terminated in Q2 2024, 12 from Tech department. "
            "Correlates with S3 enterprise churn."
        ),
        "count": 15,
        "dept_name": "Công nghệ",
        "period_start": date(2024, 4, 1),
        "period_end": date(2024, 6, 30),
    },
    "H3": {
        "name": "Salary inversion — 3 manager/report pairs",
        "domain": "hr",
        "type": "warning",
        "description": (
            "3 pairs where manager.salary < their Senior direct report.salary."
        ),
        "count": 3,
    },
    "H4": {
        "name": "Overtime surge Sales team March 2024",
        "domain": "hr",
        "type": "positive",
        "description": (
            "payroll.overtime_hours for Sales dept employees averages 4× "
            "baseline in March 2024. Directly correlated with S1 revenue spike."
        ),
        "multiplier": 4.0,
        "dept_name": "Kinh doanh",
        "month": 3,
        "year": 2024,
    },
    "H5": {
        "name": "Budget overrun Finance dept Q3 2024",
        "domain": "hr",
        "type": "negative",
        "description": (
            "SUM(payroll.net_salary) for Finance > departments.budget "
            "for 3 consecutive months in Q3 2024."
        ),
        "dept_name": "Tài chính",
        "months": [(7, 2024), (8, 2024), (9, 2024)],
    },
}

# =============================================================================
# HELPERS
# =============================================================================

def dec(value: float, places: int = 2) -> Decimal:
    """Round float → Decimal with given decimal places."""
    quantize_str = "0." + "0" * places
    return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def weighted_choice(weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = list(weights.values())
    return random.choices(keys, weights=probs, k=1)[0]


def lognormal_amount(mean_vnd: float, sigma: float = 0.6) -> Decimal:
    """Sample from lognormal — realistic for revenue distributions."""
    val = np.random.lognormal(mean=math.log(mean_vnd), sigma=sigma)
    return dec(max(val, 50_000))


def execute_many(cur, sql: str, rows: list[dict]) -> None:
    psycopg2.extras.execute_values(
        cur, sql, rows, template=None, page_size=500
    )


# =============================================================================
# DOMAIN 1 — SALES
# =============================================================================

def seed_customers(cur) -> list[dict]:
    """600 customers with region distribution matching REGION_WEIGHTS."""
    segments = ["Enterprise", "SME", "Retail", "Government"]
    seg_weights = [0.15, 0.35, 0.40, 0.10]

    CITIES = {
        "Miền Bắc":   ["Hà Nội", "Hải Phòng", "Quảng Ninh", "Nam Định", "Thái Nguyên"],
        "Miền Trung": ["Đà Nẵng", "Huế", "Đà Lạt", "Nha Trang", "Quy Nhơn"],
        "Miền Nam":   ["TP.HCM", "Cần Thơ", "Biên Hòa", "Vũng Tàu", "Long An"],
        "Tây Nguyên": ["Buôn Ma Thuột", "Pleiku", "Kon Tum", "Gia Nghĩa"],
    }

    rows = []
    regions_list = random.choices(
        REGIONS,
        weights=[REGION_WEIGHTS[r] for r in REGIONS],
        k=600,
    )
    # Mark 20 customers as churned (anomaly S3) — Enterprise, had orders in 2023
    churn_indices = set(random.sample(
        [i for i, r in enumerate(regions_list) if r == "Miền Bắc"],
        k=min(20, regions_list.count("Miền Bắc"))
    ))

    emails_used: set[str] = set()
    for i, region in enumerate(regions_list):
        segment = random.choices(segments, weights=seg_weights, k=1)[0]
        # Force churned customers to be Enterprise
        if i in churn_indices:
            segment = "Enterprise"
        city = random.choice(CITIES[region])
        email = fake.email()
        while email in emails_used:
            email = fake.email()
        emails_used.add(email)
        rows.append({
            "name": fake.name(),
            "email": email,
            "phone": fake.phone_number(),
            "city": city,
            "region": region,
            "segment": segment,
            "_is_churned": i in churn_indices,  # internal flag, not a DB column
        })

    sql = """
        INSERT INTO customers (name, email, phone, city, region, segment)
        VALUES %s RETURNING id, region, segment
    """
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO customers (name, email, phone, city, region, segment)
           VALUES %s""",
        [(r["name"], r["email"], r["phone"], r["city"], r["region"], r["segment"])
         for r in rows],
    )
    cur.execute("SELECT id, region, segment FROM customers ORDER BY id")
    db_rows = cur.fetchall()
    for i, db_row in enumerate(db_rows):
        rows[i]["id"] = db_row[0]
        rows[i]["_db_region"] = db_row[1]

    print(f"  ✓ customers: {len(rows)} rows")
    return rows


def seed_products(cur) -> list[dict]:
    """80 products across 6 categories. ELEC-007 is the negative-margin anomaly."""
    categories = [
        ("Electronics",    30, 0.35),   
        ("Office Supplies", 15, 0.45),
        ("Furniture",       12, 0.40),
        ("Industrial",      10, 0.30),
        ("Food & Beverage",  8, 0.50),
        ("Apparel",          5, 0.55),
    ]
    PRODUCT_NAMES = {
        "Electronics":     ["Laptop", "Màn hình", "Bàn phím cơ", "Chuột không dây",
                             "Tai nghe", "Webcam", "Hub USB-C", "Máy chiếu",
                             "Điện thoại", "Máy tính bảng"],
        "Office Supplies": ["Giấy A4", "Bút bi", "Mực in", "Folder", "Ghim bấm"],
        "Furniture":       ["Bàn làm việc", "Ghế văn phòng", "Tủ tài liệu",
                             "Kệ sách", "Bảng trắng"],
        "Industrial":      ["Găng tay bảo hộ", "Máy hàn mini", "Thước kẹp",
                             "Đèn công nghiệp", "Máy khoan"],
        "Food & Beverage": ["Cà phê hòa tan", "Trà túi lọc", "Nước uống",
                             "Bánh kẹo văn phòng"],
        "Apparel":         ["Đồng phục công ty", "Áo khoác", "Giày bảo hộ"],
    }

    rows = []
    sku_counter = {"Electronics": 0, "Office Supplies": 0, "Furniture": 0,
                   "Industrial": 0, "Food & Beverage": 0, "Apparel": 0}
    prefix = {"Electronics": "ELEC", "Office Supplies": "OFFC", "Furniture": "FURN",
               "Industrial": "INDS", "Food & Beverage": "FOOD", "Apparel": "APRL"}

    for cat_name, count, base_margin in categories:
        names_pool = PRODUCT_NAMES.get(cat_name, ["Sản phẩm"])
        for j in range(count):
            sku_counter[cat_name] += 1
            sku = f"{prefix[cat_name]}-{sku_counter[cat_name]:03d}"
            pname = f"{random.choice(names_pool)} {fake.numerify('##')}"
            cost = dec(np.random.uniform(100_000, 8_000_000))
            # Normal products: margin 15–60%
            margin = np.random.uniform(base_margin - 0.10, base_margin + 0.15)
            unit_price = dec(float(cost) * (1 + margin))
            rows.append({
                "name": pname,
                "sku": sku,
                "category": cat_name,
                "unit_price": unit_price,
                "cost": cost,
                "is_active": True,
            })

    # ── Anomaly S2: ELEC-007 — normal price initially, discount applied in orders
    # The product itself has unit_price > cost (schema constraint satisfied).
    # The negative margin is realised via high discount_rate in orders.
    elec_007 = next(r for r in rows if r["sku"] == "ELEC-007")
    print(f"    [S2] ELEC-007 base price={elec_007['unit_price']}, "
          f"cost={elec_007['cost']} — discount applied at order level")

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO products (name, sku, category, unit_price, cost, is_active)
           VALUES %s""",
        [(r["name"], r["sku"], r["category"], r["unit_price"], r["cost"],
          r["is_active"]) for r in rows],
    )
    cur.execute("SELECT id, sku, category, unit_price, cost FROM products ORDER BY id")
    for i, db in enumerate(cur.fetchall()):
        rows[i].update({"id": db[0], "sku": db[1], "category": db[2],
                        "unit_price": db[3], "cost": db[4]})

    print(f"  ✓ products: {len(rows)} rows")
    return rows


def seed_orders(cur, customers: list[dict], products: list[dict]) -> list[dict]:
    """
    ~30,000 orders, 2022–2024.
    Applies: S1 (spike), S2 (discount), S3 (churn), S4 (refund), S5 (payment).
    Returns list of completed orders for cross-domain movement generation.
    """
    churned_ids = {c["id"] for c in customers if c.get("_is_churned")}
    elec_007 = next(p for p in products if p["sku"] == "ELEC-007")
    product_by_id = {p["id"]: p for p in products}
    customer_by_id = {c["id"]: c for c in customers}

    PAYMENT_METHODS = ["bank_transfer", "cash", "credit_card", "e_wallet"]
    STATUSES = ["completed", "pending", "processing", "cancelled", "refunded"]

    rows = []
    target = 30_000

    # Pre-generate dates proportionally
    dates = []
    for _ in range(target):
        d = random_date(DATE_START, DATE_END)
        dates.append(d)
    dates.sort()

    for order_date in dates:
        month = order_date.month
        year = order_date.year

        # Pick customer
        customer = random.choice(customers)
        cid = customer["id"]
        region = customer["_db_region"]
        segment = customer["segment"]

        # Anomaly S3: churned Enterprise customers place no orders after churn_date
        if cid in churned_ids and order_date >= ANOMALY_CATALOGUE["S3"]["churn_date"]:
            continue

        # Pick product
        product = random.choice(products)
        pid = product["id"]
        category = product["category"]

        quantity = int(np.random.lognormal(mean=1.5, sigma=0.8))
        quantity = max(1, min(quantity, 50))

        unit_price = product["unit_price"]
        discount_rate = Decimal("0.00")

        # Anomaly S2: ELEC-007 gets high discount from Q2 2024
        if pid == elec_007["id"] and order_date >= ANOMALY_CATALOGUE["S2"]["start_date"]:
            discount_rate = ANOMALY_CATALOGUE["S2"]["discount_rate"]

        amount_base = float(unit_price) * quantity * (1 - float(discount_rate))

        # Seasonal multiplier
        amount = amount_base * SEASONAL[month]["sales"]

        # Anomaly S1: campaign spike Miền Bắc March 2024 Enterprise
        if (region == "Miền Bắc" and year == 2024 and month == 3
                and segment == "Enterprise"):
            amount *= ANOMALY_CATALOGUE["S1"]["multiplier"]
            quantity = int(quantity * 2.5)

        # Status
        status = "completed"
        status_roll = random.random()

        # Anomaly S4: Electronics refund surge Nov 2023
        if (category == "Electronics" and year == ANOMALY_CATALOGUE["S4"]["year"]
                and month == ANOMALY_CATALOGUE["S4"]["month"]):
            refund_prob = 0.07 * ANOMALY_CATALOGUE["S4"]["multiplier"]
        else:
            refund_prob = 0.07

        if status_roll < 0.10:
            status = "cancelled"
        elif status_roll < 0.10 + refund_prob:
            status = "refunded"
        elif status_roll < 0.10 + refund_prob + 0.08:
            status = "pending"
        elif status_roll < 0.10 + refund_prob + 0.08 + 0.05:
            status = "processing"

        # Payment method
        # Anomaly S5: e_wallet surge from Q2 2024
        if order_date >= ANOMALY_CATALOGUE["S5"]["start_date"]:
            pay_weights = [0.35, 0.10, 0.20, 0.35]  # e_wallet ~35%
        else:
            pay_weights = [0.55, 0.20, 0.20, 0.05]  # e_wallet ~5%
        payment_method = random.choices(PAYMENT_METHODS, weights=pay_weights, k=1)[0]

        rows.append({
            "customer_id":  cid,
            "product_id":   pid,
            "region":       region,
            "quantity":     quantity,
            "unit_price":   unit_price,
            "discount_rate": discount_rate,
            "amount":       dec(max(amount, 0)),
            "order_date":   order_date,
            "status":       status,
            "payment_method": payment_method,
        })

    # Bulk insert — quarter/year are GENERATED ALWAYS, do NOT insert them
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO orders
               (customer_id, product_id, region, quantity, unit_price,
                discount_rate, amount, order_date, status, payment_method)
           VALUES %s""",
        [(r["customer_id"], r["product_id"], r["region"], r["quantity"],
          r["unit_price"], r["discount_rate"], r["amount"], r["order_date"],
          r["status"], r["payment_method"]) for r in rows],
    )

    # Fetch back IDs for cross-domain movement generation
    cur.execute("""
        SELECT id, product_id, region, quantity, order_date, status
        FROM orders WHERE status IN ('completed', 'processing')
        ORDER BY id
    """)
    completed = [
        {"id": r[0], "product_id": r[1], "region": r[2],
         "quantity": r[3], "order_date": r[4], "status": r[5]}
        for r in cur.fetchall()
    ]

    print(f"  ✓ orders: {len(rows)} rows ({len(completed)} completed/processing)")
    return completed


# =============================================================================
# DOMAIN 2 — INVENTORY
# =============================================================================

def seed_warehouses(cur) -> list[dict]:
    """8 warehouses — Miền Bắc gets 3 (largest region)."""
    warehouses = [
        # Miền Bắc — 3 warehouses
        {"name": "Kho Hà Nội Trung Tâm", "city": "Hà Nội",     "region": "Miền Bắc",   "capacity": 50_000},
        {"name": "Kho Hải Phòng",         "city": "Hải Phòng",  "region": "Miền Bắc",   "capacity": 30_000},
        {"name": "Kho Bắc Ninh",           "city": "Bắc Ninh",   "region": "Miền Bắc",   "capacity": 20_000},
        # Miền Trung — 1
        {"name": "Kho Đà Nẵng",            "city": "Đà Nẵng",    "region": "Miền Trung", "capacity": 25_000},
        # Miền Nam — 2
        {"name": "Kho HCM Bình Dương",     "city": "TP.HCM",     "region": "Miền Nam",   "capacity": 45_000},
        {"name": "Kho Cần Thơ",            "city": "Cần Thơ",    "region": "Miền Nam",   "capacity": 20_000},
        # Tây Nguyên — 2
        {"name": "Kho BMT",                "city": "Buôn Ma Thuột", "region": "Tây Nguyên", "capacity": 15_000},
        {"name": "Kho Pleiku",             "city": "Pleiku",     "region": "Tây Nguyên", "capacity": 10_000},
    ]
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO warehouses (name, city, region, capacity)
           VALUES %s""",
        [(w["name"], w["city"], w["region"], w["capacity"]) for w in warehouses],
    )
    cur.execute("SELECT id, name, region FROM warehouses ORDER BY id")
    for i, r in enumerate(cur.fetchall()):
        warehouses[i]["id"] = r[0]

    print(f"  ✓ warehouses: {len(warehouses)} rows")
    return warehouses


def seed_stock(
    cur,
    products: list[dict],
    warehouses: list[dict],
    completed_orders: list[dict],
) -> tuple[dict, set[int]]:
    """
    ~480 (product, warehouse) pairs — not every product in every warehouse.
    Anomalies I1 (stockout risk) and I2 (dead stock) are injected here.
    Returns:
      - stock_map: dict[(product_id, warehouse_id)] -> stock_id
      - dead_stock_ids: set[product_id] used by I2
    """
    wh_ids = [w["id"] for w in warehouses]
    wh_by_region = {}
    for w in warehouses:
        wh_by_region.setdefault(w["region"], []).append(w["id"])

    # Dead-stock SKUs (anomaly I2): choose products with no completed orders
    # in the most recent 180-day window to preserve cross-domain consistency.
    recent_cutoff = DATE_END - timedelta(days=180)
    recent_order_products = {
        o["product_id"] for o in completed_orders if o["order_date"] >= recent_cutoff
    }
    dead_stock_candidates = [p["id"] for p in products if p["id"] not in recent_order_products]
    if len(dead_stock_candidates) >= 8:
        dead_stock_ids = set(random.sample(dead_stock_candidates, k=8))
    else:
        fallback_pool = [p["id"] for p in products if p["id"] not in dead_stock_candidates]
        fill_n = max(0, 8 - len(dead_stock_candidates))
        dead_stock_ids = set(dead_stock_candidates + random.sample(fallback_pool, k=fill_n))

    # Stockout SKUs (anomaly I1): 6 products get quantity < min_threshold
    stockout_ids = set(random.sample(
        [p["id"] for p in products if p["id"] not in dead_stock_ids], k=6
    ))

    rows = []
    stock_map: dict[tuple, int] = {}

    for product in products:
        pid = product["id"]
        # Each product appears in 4–7 warehouses
        n_wh = random.randint(4, min(7, len(wh_ids)))
        selected_wh = random.sample(wh_ids, k=n_wh)

        for wh_id in selected_wh:
            min_threshold = random.randint(50, 300)

            if pid in stockout_ids:
                # Anomaly I1: quantity below min_threshold
                quantity = random.randint(0, min_threshold - 1)
            elif pid in dead_stock_ids:
                # Anomaly I2: high quantity (overstock), no movement for 180+ days
                quantity = random.randint(500, 2000)
            else:
                quantity = random.randint(min_threshold, min_threshold * 10)

            unit_cost = dec(float(product["cost"]) * random.uniform(0.95, 1.05))
            reorder_qty = min_threshold * 3

            rows.append({
                "product_id":    pid,
                "warehouse_id":  wh_id,
                "quantity":      quantity,
                "min_threshold": min_threshold,
                "reorder_qty":   reorder_qty,
                "unit_cost":     unit_cost,
            })

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO stock
               (product_id, warehouse_id, quantity, min_threshold, reorder_qty, unit_cost)
           VALUES %s""",
        [(r["product_id"], r["warehouse_id"], r["quantity"], r["min_threshold"],
          r["reorder_qty"], r["unit_cost"]) for r in rows],
    )
    cur.execute("SELECT id, product_id, warehouse_id FROM stock ORDER BY id")
    for db in cur.fetchall():
        stock_map[(db[1], db[2])] = db[0]

    print(f"  ✓ stock: {len(rows)} rows "
          f"({len(stockout_ids)} stockout, {len(dead_stock_ids)} dead-stock)")
    return stock_map, dead_stock_ids


def _nearest_warehouse(region: str, warehouses: list[dict]) -> int:
    """Return warehouse ID in the same region, or any warehouse if none."""
    same_region = [w["id"] for w in warehouses if w["region"] == region]
    return random.choice(same_region) if same_region else warehouses[0]["id"]


def seed_stock_movements(
    cur,
    products: list[dict],
    warehouses: list[dict],
    completed_orders: list[dict],
    dead_stock_ids: set[int],
) -> None:
    """
    12,000 movements total:
      - outbound: one per completed order (cross-domain link via reference_order_id)
      - inbound:  replenishment movements
      - transfer: inter-warehouse transfers
      - anomalies I3 (write-off spike), I4 (mismatch), I5 (transfer loop)
    """
    rows = []
    wh_ids = [w["id"] for w in warehouses]
    tay_nguyen_wh = [w["id"] for w in warehouses if w["region"] == "Tây Nguyên"]

    # ── Outbound: one per completed order (cross-domain) ──────────────────────
    # Safety guard for I2: keep dead_stock SKUs free of recent outbound activity.
    recent_cutoff = DATE_END - timedelta(days=180)
    skipped_recent_dead_stock = 0
    for order in completed_orders:
        if order["product_id"] in dead_stock_ids and order["order_date"] >= recent_cutoff:
            skipped_recent_dead_stock += 1
            continue
        wh_id = _nearest_warehouse(order["region"], warehouses)
        rows.append({
            "product_id":        order["product_id"],
            "from_warehouse_id": wh_id,
            "to_warehouse_id":   None,
            "quantity":          order["quantity"],
            "movement_date":     order["order_date"],
            "movement_type":     "outbound",
            "reference_order_id": order["id"],
            "note":              None,
        })

    # ── Inbound: replenishment ─────────────────────────────────────────────────
    n_inbound = 4000
    for _ in range(n_inbound):
        product = random.choice(products)
        wh_id = random.choice(wh_ids)
        movement_date = random_date(DATE_START, DATE_END)
        month = movement_date.month
        quantity = int(np.random.uniform(50, 500)
                       * SEASONAL[month]["movements"])
        rows.append({
            "product_id":        product["id"],
            "from_warehouse_id": None,
            "to_warehouse_id":   wh_id,
            "quantity":          max(1, quantity),
            "movement_date":     movement_date,
            "movement_type":     "inbound",
            "reference_order_id": None,
            "note":              None,
        })

    # ── Transfers ─────────────────────────────────────────────────────────────
    for _ in range(1200):
        from_wh, to_wh = random.sample(wh_ids, 2)
        product = random.choice(products)
        rows.append({
            "product_id":        product["id"],
            "from_warehouse_id": from_wh,
            "to_warehouse_id":   to_wh,
            "quantity":          random.randint(10, 200),
            "movement_date":     random_date(DATE_START, DATE_END),
            "movement_type":     "transfer",
            "reference_order_id": None,
            "note":              None,
        })

    # ── Anomaly I3: write-off spike Q4 2023 Tây Nguyên ────────────────────────
    if tay_nguyen_wh:
        for _ in range(120):   # 5× baseline (~24 normally)
            product = random.choice(products)
            d = random_date(date(2023, 10, 1), date(2023, 12, 31))
            rows.append({
                "product_id":        product["id"],
                "from_warehouse_id": random.choice(tay_nguyen_wh),
                "to_warehouse_id":   None,
                "quantity":          random.randint(5, 100),
                "movement_date":     d,
                "movement_type":     "write_off",
                "reference_order_id": None,
                "note":              "Bão tháng 10-12/2023",
            })
        print(f"    [I3] Injected write-off spike for Tây Nguyên Q4/2023")

    # ── Anomaly I5: transfer loop Bắc ↔ Trung ─────────────────────────────────
    bac_wh  = [w["id"] for w in warehouses if w["region"] == "Miền Bắc"]
    trung_wh = [w["id"] for w in warehouses if w["region"] == "Miền Trung"]
    if bac_wh and trung_wh:
        loop_products = random.sample(products, k=5)
        loop_start = date(2024, 2, 1)
        for lp in loop_products:
            # Bắc → Trung
            rows.append({
                "product_id":        lp["id"],
                "from_warehouse_id": random.choice(bac_wh),
                "to_warehouse_id":   random.choice(trung_wh),
                "quantity":          random.randint(50, 200),
                "movement_date":     loop_start,
                "movement_type":     "transfer",
                "reference_order_id": None,
                "note":              "Điều phối Q1/2024",
            })
            # Trung → Bắc (within 30 days — the loop)
            rows.append({
                "product_id":        lp["id"],
                "from_warehouse_id": random.choice(trung_wh),
                "to_warehouse_id":   random.choice(bac_wh),
                "quantity":          random.randint(50, 200),
                "movement_date":     loop_start + timedelta(days=random.randint(10, 28)),
                "movement_type":     "transfer",
                "reference_order_id": None,
                "note":              "Trả về kho gốc",
            })
        print(f"    [I5] Injected transfer loop (Bắc↔Trung) for 5 SKUs")

    # ── Anomaly I4: Miền Bắc Q4/2024 outbound surge without matching inbound ──
    # This creates deliberate inventory-sales mismatch for cross-domain analysis.
    if bac_wh:
        non_dead_products = [p for p in products if p["id"] not in dead_stock_ids]
        for _ in range(240):
            product = random.choice(non_dead_products) if non_dead_products else random.choice(products)
            rows.append({
                "product_id":        product["id"],
                "from_warehouse_id": random.choice(bac_wh),
                "to_warehouse_id":   None,
                "quantity":          random.randint(8, 60),
                "movement_date":     random_date(date(2024, 10, 1), date(2024, 12, 31)),
                "movement_type":     "outbound",
                "reference_order_id": None,
                "note":              "I4: outbound surge without matching inbound",
            })
        print("    [I4] Injected Miền Bắc Q4/2024 outbound-only surge")

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO stock_movements
               (product_id, from_warehouse_id, to_warehouse_id, quantity,
                movement_date, movement_type, reference_order_id, note)
           VALUES %s""",
        [(r["product_id"], r["from_warehouse_id"], r["to_warehouse_id"],
          r["quantity"], r["movement_date"], r["movement_type"],
          r["reference_order_id"], r["note"]) for r in rows],
    )
    print(f"  ✓ stock_movements: {len(rows)} rows "
          f"(I2 recent-outbound skipped: {skipped_recent_dead_stock})")


# =============================================================================
# DOMAIN 3 — HR
# =============================================================================

DEPT_DEFINITIONS = [
    # (name, region, budget_VND, target_size)
    ("Kinh doanh",      "Miền Bắc",   5_000_000_000, 45),
    ("Công nghệ",       "HQ",          4_500_000_000, 40),
    ("Vận hành",        "Miền Nam",    3_000_000_000, 35),
    ("Tài chính",       "HQ",          4_800_000_000, 20),
    ("Nhân sự",         "HQ",          1_500_000_000, 15),
    ("Marketing",       "Miền Bắc",   2_500_000_000, 25),
    ("Logistics",       "Miền Nam",    2_000_000_000, 30),
    ("Pháp chế",        "HQ",          1_200_000_000, 10),
    ("R&D",             "Miền Bắc",   3_500_000_000, 20),
    ("Ban giám đốc",    "HQ",          1_000_000_000, 10),
]

LEVEL_SALARY_RANGES = {
    "Junior":    (7_000_000,  12_000_000),
    "Mid":       (13_000_000, 20_000_000),
    "Senior":    (21_000_000, 32_000_000),
    "Lead":      (33_000_000, 45_000_000),
    "Manager":   (46_000_000, 65_000_000),
    "Director":  (66_000_000, 100_000_000),
    "C-Level":   (101_000_000, 200_000_000),
}

LEVEL_WEIGHTS = {
    "Junior": 0.35, "Mid": 0.30, "Senior": 0.20,
    "Lead": 0.08, "Manager": 0.05, "Director": 0.015, "C-Level": 0.005,
}


def seed_departments(cur) -> list[dict]:
    """Insert 10 departments without manager_id (set after employees)."""
    rows = []
    for name, region, budget, _ in DEPT_DEFINITIONS:
        rows.append({
            "name": name, "region": region,
            "budget": dec(budget), "headcount": 0,
            "manager_id": None,
        })
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO departments (name, region, budget)
           VALUES %s""",
        [(r["name"], r["region"], r["budget"]) for r in rows],
    )
    cur.execute("SELECT id, name, region FROM departments ORDER BY id")
    for i, db in enumerate(cur.fetchall()):
        rows[i]["id"] = db[0]
        rows[i]["name"] = db[1]

    print(f"  ✓ departments: {len(rows)} rows (manager_id set after employees)")
    return rows


def seed_employees(cur, departments: list[dict]) -> list[dict]:
    """
    250 employees across 10 departments.
    Anomalies H2 (attrition cluster Tech) and H3 (salary inversion) injected here.
    """
    dept_map = {d["name"]: d for d in departments}
    emails_used: set[str] = set()

    # DEPT_DEFINITIONS drives target headcount
    dept_target = {name: size for name, _, _, size in DEPT_DEFINITIONS}

    ROLES_BY_DEPT = {
        "Kinh doanh":    ["Sales Executive", "Account Manager", "Sales Lead", "Sales Director"],
        "Công nghệ":     ["Software Engineer", "DevOps Engineer", "Tech Lead", "CTO"],
        "Vận hành":      ["Operations Analyst", "Operations Manager", "COO"],
        "Tài chính":     ["Accountant", "Financial Analyst", "CFO"],
        "Nhân sự":       ["HR Specialist", "Recruiter", "HR Manager", "CHRO"],
        "Marketing":     ["Marketing Specialist", "Content Creator", "Marketing Manager", "CMO"],
        "Logistics":     ["Logistics Coordinator", "Warehouse Supervisor", "Logistics Manager"],
        "Pháp chế":      ["Legal Counsel", "Compliance Officer", "Legal Director"],
        "R&D":           ["Research Engineer", "Data Scientist", "R&D Lead", "Chief Scientist"],
        "Ban giám đốc":  ["CEO", "Executive Assistant", "Board Member"],
    }

    rows = []
    for dept in departments:
        dname = dept["name"]
        target = dept_target.get(dname, 20)
        for _ in range(target):
            level = weighted_choice(LEVEL_WEIGHTS)
            sal_min, sal_max = LEVEL_SALARY_RANGES[level]
            salary = dec(np.random.uniform(sal_min, sal_max))

            hire_date = random_date(date(2018, 1, 1), date(2023, 12, 31))
            status = "active"
            end_date = None

            email = fake.email()
            while email in emails_used:
                email = fake.email()
            emails_used.add(email)

            perf = round(np.random.beta(a=3, b=1.5) * 4 + 1, 1)  # skewed toward 3-5
            perf = min(5.0, max(1.0, perf))

            role = random.choice(ROLES_BY_DEPT.get(dname, ["Chuyên viên"]))

            rows.append({
                "name":            fake.name(),
                "email":           email,
                "department_id":   dept["id"],
                "role":            role,
                "level":           level,
                "salary":          salary,
                "hire_date":       hire_date,
                "end_date":        end_date,
                "status":          status,
                "performance_score": dec(perf, 1),
            })

    # ── Anomaly H2: attrition cluster in Tech dept Q2 2024 ────────────────────
    tech_employees = [r for r in rows if r["department_id"] == dept_map["Công nghệ"]["id"]]
    terminated = random.sample(tech_employees, k=min(12, len(tech_employees)))
    for emp in terminated:
        emp["status"] = "terminated"
        emp["end_date"] = random_date(date(2024, 4, 1), date(2024, 6, 30))
    # 3 more from other depts
    others = [r for r in rows if r not in tech_employees]
    for emp in random.sample(others, k=min(3, len(others))):
        emp["status"] = "terminated"
        emp["end_date"] = random_date(date(2024, 4, 1), date(2024, 6, 30))
    print(f"    [H2] Injected attrition cluster: 15 terminated in Q2/2024 "
          f"(12 from Công nghệ)")

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO employees
               (name, email, department_id, role, level, salary,
                hire_date, end_date, status, performance_score)
           VALUES %s""",
        [(r["name"], r["email"], r["department_id"], r["role"], r["level"],
          r["salary"], r["hire_date"], r["end_date"], r["status"],
          r["performance_score"]) for r in rows],
    )
    cur.execute("""
        SELECT id, department_id, level, salary, status, performance_score
        FROM employees ORDER BY id
    """)
    for i, db in enumerate(cur.fetchall()):
        rows[i].update({
            "id": db[0], "department_id": db[1], "level": db[2],
            "salary": db[3], "status": db[4], "performance_score": db[5],
        })

    # ── Anomaly H3: salary inversion — set manager salary < senior report ─────
    seniors = [r for r in rows if r["level"] == "Senior"]
    managers = [r for r in rows if r["level"] == "Manager"]
    inverted = 0
    for mgr in random.sample(managers, k=min(3, len(managers))):
        senior = random.choice(seniors)
        if float(senior["salary"]) > float(mgr["salary"]):
            continue  # already inverted naturally
        # Force inversion: manager gets salary slightly below the senior
        new_mgr_salary = dec(float(senior["salary"]) * random.uniform(0.85, 0.97))
        cur.execute(
            "UPDATE employees SET salary = %s WHERE id = %s",
            (new_mgr_salary, mgr["id"])
        )
        mgr["salary"] = new_mgr_salary
        inverted += 1
    print(f"    [H3] Salary inversion injected for {inverted} manager/senior pairs")

    # ── Wire department managers ───────────────────────────────────────────────
    for dept in departments:
        leads = [r for r in rows
                 if r["department_id"] == dept["id"]
                 and r["level"] in ("Manager", "Director", "C-Level")
                 and r["status"] == "active"]
        if leads:
            mgr = random.choice(leads)
            cur.execute(
                "UPDATE departments SET manager_id = %s WHERE id = %s",
                (mgr["id"], dept["id"])
            )
    print(f"  ✓ employees: {len(rows)} rows")
    return rows


def seed_payroll(cur, employees: list[dict], departments: list[dict]) -> None:
    """
    2 years of monthly payroll (2023–2024).
    Anomalies H1 (bonus outliers), H4 (overtime surge), H5 (budget overrun).
    """
    dept_map = {d["id"]: d for d in departments}
    sales_dept_id = next(d["id"] for d in departments if d["name"] == "Kinh doanh")
    finance_dept_id = next(d["id"] for d in departments if d["name"] == "Tài chính")
    finance_dept = dept_map[finance_dept_id]

    # Identify bonus outlier employees (H1): 4 employees
    bonus_outlier_ids = set(
        r["id"] for r in random.sample(employees, k=4)
    )
    # 2 of them have high performance, 2 don't (justified vs unjustified)
    justified_bonus_ids = set(
        r["id"] for r in sorted(
            [e for e in employees if e["id"] in bonus_outlier_ids],
            key=lambda x: float(x["performance_score"]),
            reverse=True
        )[:2]
    )

    rows = []
    for emp in employees:
        eid = emp["id"]
        base = float(emp["salary"])
        dept_id = emp["department_id"]

        for year in (2023, 2024):
            for month in range(1, 13):
                # Skip months after termination
                if emp["status"] == "terminated" and emp["end_date"]:
                    end = emp["end_date"]
                    if (year > end.year or
                            (year == end.year and month > end.month)):
                        continue

                # Normal bonus: 0–20% of base
                bonus_pct = np.random.beta(a=1.5, b=5) * 0.20
                bonus = base * bonus_pct

                # Anomaly H1: outlier bonus
                if eid in bonus_outlier_ids and month == 12:
                    bonus = base * random.uniform(3.0, 5.0)

                # Anomaly H4: overtime surge Sales March 2024
                if (dept_id == sales_dept_id and year == 2024 and month == 3):
                    overtime = round(
                        np.random.uniform(40, 80)
                        * ANOMALY_CATALOGUE["H4"]["multiplier"], 1
                    )
                else:
                    overtime = round(
                        np.random.uniform(0, 20)
                        * SEASONAL[month]["overtime"], 1
                    )

                deduction = base * np.random.uniform(0.08, 0.12)  # social insurance etc.

                # Anomaly H5: Finance budget overrun Q3 2024
                # Inflate salaries slightly to push dept over budget
                if (dept_id == finance_dept_id and year == 2024
                        and month in (7, 8, 9)):
                    base_for_month = base * 1.18
                else:
                    base_for_month = base

                net = base_for_month + bonus - deduction
                paid_at = date(year, month, random.randint(25, 28))

                rows.append({
                    "employee_id":   eid,
                    "month":         month,
                    "year":          year,
                    "base_salary":   dec(base_for_month),
                    "bonus":         dec(bonus),
                    "deduction":     dec(deduction),
                    "overtime_hours": overtime,
                    "net_salary":    dec(max(net, 1)),
                    "paid_at":       paid_at,
                })

    # Validate net_salary constraint: net = base + bonus - deduction
    for r in rows:
        expected = float(r["base_salary"]) + float(r["bonus"]) - float(r["deduction"])
        r["net_salary"] = dec(max(expected, 1))

    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO payroll
               (employee_id, month, year, base_salary, bonus, deduction,
                overtime_hours, net_salary, paid_at)
           VALUES %s""",
        [(r["employee_id"], r["month"], r["year"], r["base_salary"],
          r["bonus"], r["deduction"], r["overtime_hours"],
          r["net_salary"], r["paid_at"]) for r in rows],
    )
    print(f"  ✓ payroll: {len(rows)} rows "
          f"[H1: {len(bonus_outlier_ids)} outlier employees, "
          f"H4: Sales overtime surge Mar/2024, "
          f"H5: Finance budget overrun Q3/2024]")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("ADBA Seed Data Generator")
    print("=" * 50)
    print(f"Connecting to: {DATABASE_URL.split('@')[-1]}")

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # ── Sales ──────────────────────────────────────────────────────────
            print("\n[1/3] Seeding Sales domain …")
            customers = seed_customers(cur)
            products  = seed_products(cur)
            completed_orders = seed_orders(cur, customers, products)

            # ── Inventory ──────────────────────────────────────────────────────
            print("\n[2/3] Seeding Inventory domain …")
            warehouses = seed_warehouses(cur)
            stock_map, dead_stock_ids = seed_stock(cur, products, warehouses, completed_orders)
            seed_stock_movements(cur, products, warehouses, completed_orders, dead_stock_ids)

            # ── HR ─────────────────────────────────────────────────────────────
            print("\n[3/3] Seeding HR domain …")
            departments = seed_departments(cur)
            employees   = seed_employees(cur, departments)
            seed_payroll(cur, employees, departments)

        conn.commit()
        print("\n✓ All domains seeded and committed.")

        # ── Quick verification ─────────────────────────────────────────────────
        print("\nVerification row counts:")
        with conn.cursor() as cur:
            tables = [
                "customers", "products", "orders",
                "warehouses", "stock", "stock_movements",
                "departments", "employees", "payroll",
            ]
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                count = cur.fetchone()[0]
                print(f"  {t:25s}: {count:>7,}")

        print("\nAnomaly summary (injected):")
        for aid, a in ANOMALY_CATALOGUE.items():
            print(f"  [{aid}] {a['name']}")

    except Exception as exc:
        conn.rollback()
        print(f"\n✗ Error — rolled back: {exc}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()