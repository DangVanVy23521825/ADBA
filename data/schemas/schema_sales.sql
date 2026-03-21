-- =============================================================================
-- SALES DOMAIN SCHEMA
-- Tables: customers, products, orders
-- Key analytical dimensions: region, category, order_date, quarter, year
-- Cross-domain join: product_id → inventory.stock, inventory.stock_movements
-- =============================================================================

-- -----------------------------------------------------------------------------
-- customers
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(150)        NOT NULL,
    email         VARCHAR(254)        NOT NULL UNIQUE,
    phone         VARCHAR(20),
    city          VARCHAR(100)        NOT NULL,
    region        VARCHAR(50)         NOT NULL
                      CHECK (region IN ('Miền Bắc', 'Miền Trung', 'Miền Nam', 'Tây Nguyên')),
    segment       VARCHAR(50)         NOT NULL
                      CHECK (segment IN ('Enterprise', 'SME', 'Retail', 'Government')),
    created_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_customers_region  ON customers (region);
CREATE INDEX idx_customers_segment ON customers (segment);

-- -----------------------------------------------------------------------------
-- products
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(200)        NOT NULL,
    sku           VARCHAR(50)         NOT NULL UNIQUE,
    category      VARCHAR(100)        NOT NULL
                      CHECK (category IN (
                          'Electronics', 'Office Supplies', 'Furniture',
                          'Industrial', 'Food & Beverage', 'Apparel'
                      )),
    unit_price    NUMERIC(12, 2)      NOT NULL CHECK (unit_price > 0),
    cost          NUMERIC(12, 2)      NOT NULL CHECK (cost > 0),
    is_active     BOOLEAN             NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_margin CHECK (unit_price > cost)
);

CREATE INDEX idx_products_category  ON products (category);
CREATE INDEX idx_products_is_active ON products (is_active);

-- -----------------------------------------------------------------------------
-- orders
-- Computed columns quarter & year are stored for fast GROUP BY in BI queries.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    customer_id     INT                 NOT NULL REFERENCES customers (id),
    product_id      INT                 NOT NULL REFERENCES products  (id),
    region          VARCHAR(50)         NOT NULL
                        CHECK (region IN ('Miền Bắc', 'Miền Trung', 'Miền Nam', 'Tây Nguyên')),
    quantity        INT                 NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(12, 2)      NOT NULL CHECK (unit_price > 0),
    discount_rate   NUMERIC(5, 4)       NOT NULL DEFAULT 0
                        CHECK (discount_rate >= 0 AND discount_rate < 1),
    amount          NUMERIC(14, 2)      NOT NULL CHECK (amount >= 0),
    order_date      DATE                NOT NULL,
    quarter         SMALLINT            NOT NULL GENERATED ALWAYS AS
                        (EXTRACT(QUARTER FROM order_date)::SMALLINT) STORED,
    year            SMALLINT            NOT NULL GENERATED ALWAYS AS
                        (EXTRACT(YEAR   FROM order_date)::SMALLINT) STORED,
    status          VARCHAR(20)         NOT NULL DEFAULT 'completed'
                        CHECK (status IN ('pending', 'processing', 'completed', 'cancelled', 'refunded')),
    payment_method  VARCHAR(30)         NOT NULL DEFAULT 'bank_transfer'
                        CHECK (payment_method IN ('bank_transfer', 'cash', 'credit_card', 'e_wallet')),
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

-- Composite indexes tuned for the most common BI query patterns
CREATE INDEX idx_orders_date_region   ON orders (order_date, region);
CREATE INDEX idx_orders_year_quarter  ON orders (year, quarter);
CREATE INDEX idx_orders_customer      ON orders (customer_id);
CREATE INDEX idx_orders_product       ON orders (product_id);
CREATE INDEX idx_orders_status        ON orders (status);

-- Partial index: exclude cancelled/refunded for revenue rollups
CREATE INDEX idx_orders_completed     ON orders (order_date, region, amount)
    WHERE status IN ('completed', 'processing');
