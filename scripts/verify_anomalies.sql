-- ============================================================
-- ADBA — Anomaly Verification Queries
-- Based on exact schema columns:
--   orders   : id, customer_id, product_id, region, quantity,
--              unit_price, discount_rate, amount, order_date,
--              quarter (GENERATED), year (GENERATED), status,
--              payment_method
--   products : id, name, sku, category, unit_price, cost
--   stock    : id, product_id, warehouse_id, quantity,
--              min_threshold, reorder_qty, unit_cost
--   stock_movements: id, product_id, from_warehouse_id,
--              to_warehouse_id, quantity, movement_date,
--              movement_type, reference_order_id, note
--   warehouses: id, name, city, region, capacity, is_active
--   employees : id, name, email, department_id, role, level,
--               salary, hire_date, end_date, status,
--               performance_score
--   departments: id, name, region, budget, headcount, manager_id
--   payroll   : id, employee_id, month, year, base_salary,
--               bonus, deduction, overtime_hours, net_salary,
--               paid_at
-- ============================================================

-- ── S1: Revenue spike tháng 3/2024 Miền Bắc ─────────────────
-- Dùng EXTRACT(MONTH FROM order_date) vì orders không có cột month
SELECT
    EXTRACT(MONTH FROM order_date)::INT AS month,
    ROUND(SUM(amount)) AS total_revenue
FROM orders
WHERE status = 'completed'
  AND region = 'Miền Bắc'
  AND year = 2024
  AND EXTRACT(MONTH FROM order_date) BETWEEN 1 AND 5
GROUP BY 1
ORDER BY 1;
-- Expect: tháng 3 cao hơn các tháng còn lại ít nhất 2×

-- ── S2: ELEC-007 effective price < cost từ Q2/2024 ───────────
SELECT
    o.order_date,
    p.sku,
    p.cost,
    o.unit_price,
    o.discount_rate,
    ROUND(o.unit_price * (1 - o.discount_rate), 0) AS effective_price,
    ROUND(o.unit_price * (1 - o.discount_rate) - p.cost, 0) AS margin
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE p.sku = 'ELEC-007'
  AND o.order_date >= '2024-04-01'
ORDER BY o.order_date
LIMIT 5;
-- Expect: margin âm (effective_price < cost)

-- ── S3: Enterprise churn — không order sau 2024-07-01 ────────
SELECT COUNT(DISTINCT c.id) AS churned_count
FROM customers c
WHERE c.segment = 'Enterprise'
  AND EXISTS (
      SELECT 1 FROM orders o
      WHERE o.customer_id = c.id
        AND o.order_date < '2024-07-01'
  )
  AND NOT EXISTS (
      SELECT 1 FROM orders o
      WHERE o.customer_id = c.id
        AND o.order_date >= '2024-07-01'
  );
-- Expect: >= 20

-- ── S4: Refund surge Electronics tháng 11/2023 ───────────────
SELECT
    EXTRACT(MONTH FROM o.order_date)::INT AS month,
    COUNT(*) AS refund_count
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.status = 'refunded'
  AND p.category = 'Electronics'
  AND o.year = 2023
GROUP BY 1
ORDER BY 1;
-- Expect: tháng 11 cao hơn các tháng khác rõ rệt (~3.4×)

-- ── S5: Payment method shift — e_wallet tăng từ Q2/2024 ──────
SELECT
    CASE WHEN order_date < '2024-04-01' THEN 'before_Q2_2024'
         ELSE 'from_Q2_2024' END AS period,
    payment_method,
    COUNT(*) AS cnt,
    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (
            PARTITION BY
                CASE WHEN order_date < '2024-04-01'
                     THEN 'before_Q2_2024' ELSE 'from_Q2_2024' END
        ), 1
    ) AS pct
FROM orders
GROUP BY 1, 2
ORDER BY 1, cnt DESC;
-- Expect: e_wallet pct: before ~5%, from_Q2_2024 ~35%

-- ── I1: Stockout — quantity < min_threshold ──────────────────
SELECT
    p.sku,
    p.name,
    s.quantity,
    s.min_threshold,
    s.quantity - s.min_threshold AS gap
FROM stock s
JOIN products p ON s.product_id = p.id
WHERE s.quantity < s.min_threshold
ORDER BY gap ASC;
-- Expect: rows từ đúng 6 SKUs khác nhau

-- ── I2: Dead stock — không có outbound trong 180 ngày ────────
-- DATE_END = 2024-12-31, cutoff = 2024-07-04
SELECT
    p.sku,
    p.name,
    MAX(sm.movement_date)   AS last_outbound,
    SUM(s.quantity)         AS total_stock_all_wh
FROM stock s
JOIN products p ON s.product_id = p.id
LEFT JOIN stock_movements sm
       ON sm.product_id = p.id
      AND sm.movement_type = 'outbound'
GROUP BY p.id, p.sku, p.name
HAVING MAX(sm.movement_date) IS NULL
    OR MAX(sm.movement_date) < '2024-07-04'
ORDER BY total_stock_all_wh DESC;
-- Expect: đúng 8 SKUs, total_stock cao (500–2000)

-- ── I3: Write-off spike Q4/2023 Tây Nguyên ───────────────────
SELECT
    w.region,
    EXTRACT(YEAR    FROM sm.movement_date)::INT    AS yr,
    EXTRACT(QUARTER FROM sm.movement_date)::INT    AS qtr,
    COUNT(*)        AS event_count,
    SUM(sm.quantity) AS total_qty
FROM stock_movements sm
JOIN warehouses w ON sm.from_warehouse_id = w.id
WHERE sm.movement_type = 'write_off'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
-- Expect: Tây Nguyên yr=2023 qtr=4 cao hơn các quý khác ~5×

-- ── I4: Mismatch — outbound Q4/2024 Miền Bắc không có order ─
SELECT
    w.region,
    COUNT(*)         AS orphan_outbound_rows,
    SUM(sm.quantity) AS total_orphan_qty
FROM stock_movements sm
JOIN warehouses w ON sm.from_warehouse_id = w.id
WHERE sm.movement_type = 'outbound'
  AND sm.reference_order_id IS NULL
  AND sm.movement_date >= '2024-10-01'
GROUP BY w.region
ORDER BY orphan_outbound_rows DESC;
-- Expect: Miền Bắc có giá trị cao nhất

-- ── I5: Transfer loop Bắc ↔ Trung ────────────────────────────
SELECT
    p.sku,
    wf.region AS from_region,
    wt.region AS to_region,
    sm.movement_date,
    sm.note
FROM stock_movements sm
JOIN warehouses wf ON sm.from_warehouse_id = wf.id
JOIN warehouses wt ON sm.to_warehouse_id   = wt.id
JOIN products   p  ON sm.product_id        = p.id
WHERE sm.note IN ('Điều phối Q1/2024', 'Trả về kho gốc')
ORDER BY p.sku, sm.movement_date;
-- Expect: 5 SKUs, mỗi SKU 2 rows: Bắc→Trung rồi Trung→Bắc

-- ── H1: Bonus outliers tháng 12 ──────────────────────────────
SELECT
    e.name,
    e.level,
    e.performance_score,
    p.year,
    p.month,
    ROUND(p.base_salary)     AS base,
    ROUND(p.bonus)           AS bonus,
    ROUND(p.bonus / p.base_salary, 2) AS bonus_ratio
FROM payroll p
JOIN employees e ON p.employee_id = e.id
WHERE p.month = 12
  AND p.bonus / p.base_salary > 2.0
ORDER BY bonus_ratio DESC;
-- Expect: 4 employees với bonus_ratio 3–5×

-- ── H2: Attrition cluster Q2/2024 Công nghệ ─────────────────
SELECT
    d.name        AS department,
    COUNT(*)      AS terminated_in_q2_2024
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.status = 'terminated'
  AND e.end_date BETWEEN '2024-04-01' AND '2024-06-30'
GROUP BY d.name
ORDER BY terminated_in_q2_2024 DESC;
-- Expect: "Công nghệ" dẫn đầu với 12

-- ── H3: Salary inversion Manager < Senior ────────────────────
SELECT
    m.name          AS manager_name,
    m.salary        AS manager_salary,
    s.name          AS senior_name,
    s.salary        AS senior_salary,
    ROUND(s.salary - m.salary, 0) AS gap
FROM employees m
JOIN employees s
  ON m.department_id = s.department_id
 AND m.level = 'Manager'
 AND s.level = 'Senior'
 AND m.salary < s.salary
ORDER BY gap DESC;
-- Expect: >= 3 pairs

-- ── H4: Overtime surge Kinh doanh tháng 3/2024 ───────────────
-- payroll có cột month (SMALLINT) — dùng trực tiếp
SELECT
    p.year,
    p.month,
    ROUND(AVG(p.overtime_hours), 1) AS avg_overtime
FROM payroll p
JOIN employees   e ON p.employee_id    = e.id
JOIN departments d ON e.department_id  = d.id
WHERE d.name = 'Kinh doanh'
  AND p.year = 2024
  AND p.month BETWEEN 1 AND 6
GROUP BY p.year, p.month
ORDER BY p.month;
-- Expect: tháng 3 cao hơn các tháng khác ~4×

-- ── H5: Budget overrun Tài chính Q3/2024 ─────────────────────
SELECT
    p.year,
    p.month,
    ROUND(SUM(p.net_salary))       AS total_payroll,
    ROUND(d.budget)                AS dept_budget,
    ROUND(SUM(p.net_salary) - d.budget) AS overrun
FROM payroll p
JOIN employees   e ON p.employee_id   = e.id
JOIN departments d ON e.department_id = d.id
WHERE d.name = 'Tài chính'
  AND p.year = 2024
GROUP BY p.year, p.month, d.budget
ORDER BY p.month;
-- Expect: tháng 7,8,9 có overrun > 0

-- ── Cross-domain: S1 + H4 causal chain ───────────────────────
-- Revenue Miền Bắc by month 2024
SELECT
    EXTRACT(MONTH FROM order_date)::INT AS month,
    ROUND(SUM(amount) / 1e6, 1)        AS revenue_million_vnd
FROM orders
WHERE region = 'Miền Bắc'
  AND year   = 2024
  AND status = 'completed'
  AND EXTRACT(MONTH FROM order_date) <= 6
GROUP BY 1
ORDER BY 1;

-- Overtime Kinh doanh by month 2024
SELECT
    p.month,
    ROUND(AVG(p.overtime_hours), 1) AS avg_overtime_hours
FROM payroll p
JOIN employees   e ON p.employee_id   = e.id
JOIN departments d ON e.department_id = d.id
WHERE d.name   = 'Kinh doanh'
  AND p.year   = 2024
  AND p.month <= 6
GROUP BY p.month
ORDER BY p.month;
-- Expect: cả 2 queries đều có tháng 3 là đỉnh