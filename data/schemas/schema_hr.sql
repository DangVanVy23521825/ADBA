-- =============================================================================
-- HR DOMAIN SCHEMA
-- Tables: departments, employees, payroll
-- Key analytical dimensions: department_id, level, hire_date, month/year
-- Cross-domain: departments.region → sales regions for workforce-vs-revenue BI
-- =============================================================================

-- -----------------------------------------------------------------------------
-- departments
-- manager_id is a self-referential FK (set after employees are inserted).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS departments (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(150)        NOT NULL UNIQUE,
    region        VARCHAR(50)         NOT NULL
                      CHECK (region IN ('Miền Bắc', 'Miền Trung', 'Miền Nam', 'Tây Nguyên', 'HQ')),
    budget        NUMERIC(16, 2)      NOT NULL CHECK (budget > 0),
    headcount     INT                 NOT NULL DEFAULT 0 CHECK (headcount >= 0),
    manager_id    INT,                                -- FK set below after employees
    created_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_departments_region ON departments (region);

-- -----------------------------------------------------------------------------
-- employees
-- level enum keeps salary band analysis clean.
-- status covers active workforce vs. attrition analysis.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150)        NOT NULL,
    email           VARCHAR(254)        NOT NULL UNIQUE,
    department_id   INT                 NOT NULL REFERENCES departments (id),
    role            VARCHAR(100)        NOT NULL,
    level           VARCHAR(20)         NOT NULL
                        CHECK (level IN ('Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director', 'C-Level')),
    salary          NUMERIC(12, 2)      NOT NULL CHECK (salary > 0),
    hire_date       DATE                NOT NULL,
    end_date        DATE,
    status          VARCHAR(20)         NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'on_leave', 'terminated', 'contractor')),
    performance_score NUMERIC(3, 1)     CHECK (performance_score BETWEEN 1.0 AND 5.0),
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_end_after_hire CHECK (end_date IS NULL OR end_date > hire_date)
);

CREATE INDEX idx_employees_dept      ON employees (department_id);
CREATE INDEX idx_employees_level     ON employees (level);
CREATE INDEX idx_employees_status    ON employees (status);
CREATE INDEX idx_employees_hire_date ON employees (hire_date);
-- Partial index: active employees only (most queries exclude terminated)
CREATE INDEX idx_employees_active    ON employees (department_id, level)
    WHERE status = 'active';

-- Wire manager FK after employees table exists
ALTER TABLE departments
    ADD CONSTRAINT fk_dept_manager
    FOREIGN KEY (manager_id) REFERENCES employees (id) DEFERRABLE INITIALLY DEFERRED;

-- Trigger: sync departments.headcount when employee status changes
CREATE OR REPLACE FUNCTION fn_dept_headcount_sync()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE departments
    SET headcount = (
        SELECT COUNT(*)
        FROM   employees
        WHERE  department_id = COALESCE(NEW.department_id, OLD.department_id)
          AND  status = 'active'
    )
    WHERE id = COALESCE(NEW.department_id, OLD.department_id);
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_headcount_sync
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION fn_dept_headcount_sync();

-- -----------------------------------------------------------------------------
-- payroll
-- Append-only monthly payroll ledger.
-- net_salary is stored (not computed) to allow historical correctness even if
-- salary changes, which is crucial for accurate YoY compensation analysis.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payroll (
    id              SERIAL PRIMARY KEY,
    employee_id     INT             NOT NULL REFERENCES employees (id),
    month           SMALLINT        NOT NULL CHECK (month BETWEEN 1 AND 12),
    year            SMALLINT        NOT NULL CHECK (year >= 2000),
    base_salary     NUMERIC(12, 2)  NOT NULL CHECK (base_salary > 0),
    bonus           NUMERIC(12, 2)  NOT NULL DEFAULT 0 CHECK (bonus >= 0),
    deduction       NUMERIC(12, 2)  NOT NULL DEFAULT 0 CHECK (deduction >= 0),
    overtime_hours  NUMERIC(5, 1)   NOT NULL DEFAULT 0 CHECK (overtime_hours >= 0),
    net_salary      NUMERIC(12, 2)  NOT NULL CHECK (net_salary > 0),
    paid_at         DATE            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (employee_id, month, year),

    CONSTRAINT chk_net_salary CHECK (
        net_salary = base_salary + bonus - deduction
    )
);

CREATE INDEX idx_payroll_employee    ON payroll (employee_id);
CREATE INDEX idx_payroll_year_month  ON payroll (year, month);
CREATE INDEX idx_payroll_paid_at     ON payroll (paid_at);
