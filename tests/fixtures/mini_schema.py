"""Schema 4 bảng dùng chung cho test. Cố ý nhỏ và có FK bắc cầu."""

from perception.schema_model import Column, Table

MINI_TABLES = (
    Table(
        name="customers",
        columns=(Column("id", "integer"), Column("name", "character varying"),
                 Column("segment", "character varying")),
        primary_key=("id",),
        row_count=500,
        description="Khách hàng doanh nghiệp và cá nhân",
    ),
    Table(
        name="products",
        columns=(Column("id", "integer"), Column("name", "character varying"),
                 Column("category", "character varying"), Column("price", "numeric")),
        primary_key=("id",),
        row_count=80,
        description="Danh mục sản phẩm bán ra",
    ),
    Table(
        name="orders",
        columns=(Column("id", "integer"), Column("customer_id", "integer"),
                 Column("product_id", "integer"), Column("amount", "numeric"),
                 Column("order_date", "date"), Column("year", "smallint", is_generated=True)),
        primary_key=("id",),
        foreign_keys={"customer_id": "customers(id)", "product_id": "products(id)"},
        row_count=29830,
        description="Đơn hàng bán cho khách",
    ),
    Table(
        name="payroll",
        columns=(Column("id", "integer"), Column("employee_id", "integer"),
                 Column("base_salary", "numeric"), Column("month", "smallint")),
        primary_key=("id",),
        row_count=5897,
        description="Bảng lương nhân viên theo tháng",
    ),
)
