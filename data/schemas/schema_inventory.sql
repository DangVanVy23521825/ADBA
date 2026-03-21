-- =============================================================================
-- INVENTORY DOMAIN SCHEMA
-- Tables: warehouses, stock, stock_movements
-- Key analytical dimensions: region, product_id, movement_type, movement_date
-- Cross-domain join: product_id → sales.products, sales.orders
-- =============================================================================

-- -----------------------------------------------------------------------------
-- warehouses
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS warehouses (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(150)        NOT NULL,
    city          VARCHAR(100)        NOT NULL,
    region        VARCHAR(50)         NOT NULL
                      CHECK (region IN ('Miền Bắc', 'Miền Trung', 'Miền Nam', 'Tây Nguyên')),
    capacity      INT                 NOT NULL CHECK (capacity > 0),
    is_active     BOOLEAN             NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_warehouses_region ON warehouses (region);

-- -----------------------------------------------------------------------------
-- stock
-- Current inventory snapshot per (product, warehouse).
-- min_threshold triggers reorder alerts (used by Insight Agent anomaly detection).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stock (
    id              SERIAL PRIMARY KEY,
    product_id      INT             NOT NULL REFERENCES products    (id),
    warehouse_id    INT             NOT NULL REFERENCES warehouses  (id),
    quantity        INT             NOT NULL DEFAULT 0 CHECK (quantity >= 0),
    min_threshold   INT             NOT NULL DEFAULT 0 CHECK (min_threshold >= 0),
    reorder_qty     INT             NOT NULL DEFAULT 0 CHECK (reorder_qty >= 0),
    unit_cost       NUMERIC(12, 2)  NOT NULL CHECK (unit_cost > 0),
    last_updated    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (product_id, warehouse_id)
);

CREATE INDEX idx_stock_product    ON stock (product_id);
CREATE INDEX idx_stock_warehouse  ON stock (warehouse_id);
-- Partial index: below-threshold items for reorder queries
CREATE INDEX idx_stock_low        ON stock (product_id, warehouse_id)
    WHERE quantity < min_threshold;

-- Trigger: auto-update last_updated on any row change
CREATE OR REPLACE FUNCTION fn_stock_touch()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.last_updated := NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_stock_touch
    BEFORE UPDATE ON stock
    FOR EACH ROW EXECUTE FUNCTION fn_stock_touch();

-- -----------------------------------------------------------------------------
-- stock_movements
-- Append-only ledger of every inventory transaction.
-- from_warehouse_id NULL  → inbound (purchase / production)
-- to_warehouse_id   NULL  → outbound (sale / write-off)
-- Both non-NULL           → internal transfer
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stock_movements (
    id                  SERIAL PRIMARY KEY,
    product_id          INT             NOT NULL REFERENCES products   (id),
    from_warehouse_id   INT                      REFERENCES warehouses (id),
    to_warehouse_id     INT                      REFERENCES warehouses (id),
    quantity            INT             NOT NULL CHECK (quantity > 0),
    movement_date       DATE            NOT NULL,
    movement_type       VARCHAR(30)     NOT NULL
                            CHECK (movement_type IN (
                                'inbound', 'outbound', 'transfer',
                                'adjustment', 'return', 'write_off'
                            )),
    reference_order_id  INT,
    note                TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_movement_direction CHECK (
        NOT (from_warehouse_id IS NULL AND to_warehouse_id IS NULL)
    )
);

CREATE INDEX idx_movements_date          ON stock_movements (movement_date);
CREATE INDEX idx_movements_product       ON stock_movements (product_id);
CREATE INDEX idx_movements_type_date     ON stock_movements (movement_type, movement_date);
CREATE INDEX idx_movements_from_wh       ON stock_movements (from_warehouse_id)
    WHERE from_warehouse_id IS NOT NULL;
CREATE INDEX idx_movements_to_wh         ON stock_movements (to_warehouse_id)
    WHERE to_warehouse_id IS NOT NULL;
