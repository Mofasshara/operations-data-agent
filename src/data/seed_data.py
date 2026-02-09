"""Generate synthetic enterprise data for the AI agent."""

import random
from datetime import datetime, timedelta
from faker import Faker
import duckdb
from pathlib import Path
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

DATABASE_PATH = os.getenv("DATABASE_PATH", "data/enterprise.duckdb")

# Enterprise-like constants
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
PRODUCT_CATEGORIES = ["Software", "Hardware", "Services", "Subscriptions", "Support"]
SALES_CHANNELS = ["Direct", "Partner", "Online", "Reseller"]
CUSTOMER_SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]

# Products with realistic pricing
PRODUCTS = [
    {"name": "Enterprise Suite Pro", "category": "Software", "base_price": 5000},
    {"name": "Cloud Platform License", "category": "Subscriptions", "base_price": 2500},
    {"name": "Data Analytics Module", "category": "Software", "base_price": 3500},
    {"name": "Security Gateway", "category": "Hardware", "base_price": 8000},
    {"name": "API Integration Pack", "category": "Services", "base_price": 1500},
    {"name": "Premium Support Plan", "category": "Support", "base_price": 1200},
    {"name": "Developer Toolkit", "category": "Software", "base_price": 800},
    {"name": "Infrastructure Monitor", "category": "Software", "base_price": 2200},
    {"name": "Backup Solution", "category": "Services", "base_price": 1800},
    {"name": "Compliance Manager", "category": "Software", "base_price": 4500},
    {"name": "Network Appliance", "category": "Hardware", "base_price": 6500},
    {"name": "Training Package", "category": "Services", "base_price": 3000},
]


def create_tables(conn: duckdb.DuckDBPyConnection):
    """Create database tables."""

    # Customers table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR PRIMARY KEY,
            company_name VARCHAR,
            industry VARCHAR,
            segment VARCHAR,
            region VARCHAR,
            country VARCHAR,
            created_date DATE,
            account_manager VARCHAR,
            annual_revenue DECIMAL(15,2),
            employee_count INTEGER
        )
    """)

    # Products table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id VARCHAR PRIMARY KEY,
            product_name VARCHAR,
            category VARCHAR,
            base_price DECIMAL(10,2),
            launch_date DATE,
            is_active BOOLEAN
        )
    """)

    # Sales table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            sale_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR,
            product_id VARCHAR,
            sale_date DATE,
            quantity INTEGER,
            unit_price DECIMAL(10,2),
            discount_percent DECIMAL(5,2),
            total_amount DECIMAL(15,2),
            region VARCHAR,
            channel VARCHAR,
            sales_rep VARCHAR
        )
    """)

    # Sales Reps table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales_reps (
            rep_id VARCHAR PRIMARY KEY,
            rep_name VARCHAR,
            region VARCHAR,
            hire_date DATE,
            quota DECIMAL(15,2),
            email VARCHAR
        )
    """)

    # Monthly Metrics table (for forecasting)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monthly_metrics (
            month_date DATE PRIMARY KEY,
            total_revenue DECIMAL(15,2),
            total_orders INTEGER,
            new_customers INTEGER,
            churned_customers INTEGER,
            avg_order_value DECIMAL(10,2)
        )
    """)


def generate_customers(conn: duckdb.DuckDBPyConnection, n: int = 500):
    """Generate fake customer data."""
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Education", "Government"]

    customers = []
    for i in range(n):
        region = random.choice(REGIONS)
        customers.append({
            "customer_id": f"CUST-{i+1:05d}",
            "company_name": fake.company(),
            "industry": random.choice(industries),
            "segment": random.choice(CUSTOMER_SEGMENTS),
            "region": region,
            "country": fake.country(),
            "created_date": fake.date_between(start_date="-5y", end_date="-30d"),
            "account_manager": fake.name(),
            "annual_revenue": round(random.uniform(100000, 50000000), 2),
            "employee_count": random.randint(10, 10000),
        })

    conn.executemany("""
        INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [tuple(c.values()) for c in customers])

    print(f"Generated {n} customers")


def generate_products(conn: duckdb.DuckDBPyConnection):
    """Generate products from predefined list."""
    products = []
    for i, p in enumerate(PRODUCTS):
        products.append({
            "product_id": f"PROD-{i+1:03d}",
            "product_name": p["name"],
            "category": p["category"],
            "base_price": p["base_price"],
            "launch_date": fake.date_between(start_date="-3y", end_date="-6m"),
            "is_active": True,
        })

    conn.executemany("""
        INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)
    """, [tuple(p.values()) for p in products])

    print(f"Generated {len(products)} products")


def generate_sales_reps(conn: duckdb.DuckDBPyConnection, n: int = 25):
    """Generate sales rep data."""
    reps = []
    for i in range(n):
        region = REGIONS[i % len(REGIONS)]
        reps.append({
            "rep_id": f"REP-{i+1:03d}",
            "rep_name": fake.name(),
            "region": region,
            "hire_date": fake.date_between(start_date="-5y", end_date="-3m"),
            "quota": round(random.uniform(500000, 2000000), 2),
            "email": fake.email(),
        })

    conn.executemany("""
        INSERT INTO sales_reps VALUES (?, ?, ?, ?, ?, ?)
    """, [tuple(r.values()) for r in reps])

    print(f"Generated {n} sales reps")


def generate_sales(conn: duckdb.DuckDBPyConnection, n: int = 5000):
    """Generate realistic sales transactions with seasonal patterns."""

    # Get existing IDs
    customers = [r[0] for r in conn.execute("SELECT customer_id FROM customers").fetchall()]
    products = conn.execute("SELECT product_id, base_price FROM products").fetchall()
    reps = [r[0] for r in conn.execute("SELECT rep_id FROM sales_reps").fetchall()]

    sales = []
    start_date = datetime.now() - timedelta(days=730)  # 2 years of data

    for i in range(n):
        sale_date = start_date + timedelta(days=random.randint(0, 730))

        # Seasonal adjustment (Q4 boost, Q1 dip)
        month = sale_date.month
        seasonal_factor = 1.0
        if month in [10, 11, 12]:  # Q4 boost
            seasonal_factor = 1.3
        elif month in [1, 2]:  # Q1 dip
            seasonal_factor = 0.8

        product = random.choice(products)
        product_id, base_price = product

        quantity = max(1, int(random.gauss(3, 2) * seasonal_factor))
        discount = random.choice([0, 5, 10, 15, 20]) if random.random() > 0.5 else 0
        unit_price = float(base_price) * (1 - discount/100)
        total = round(unit_price * quantity, 2)

        customer = random.choice(customers)
        region = conn.execute(f"SELECT region FROM customers WHERE customer_id = '{customer}'").fetchone()[0]

        sales.append({
            "sale_id": f"SALE-{i+1:06d}",
            "customer_id": customer,
            "product_id": product_id,
            "sale_date": sale_date.date(),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_percent": discount,
            "total_amount": total,
            "region": region,
            "channel": random.choice(SALES_CHANNELS),
            "sales_rep": random.choice(reps),
        })

    conn.executemany("""
        INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [tuple(s.values()) for s in sales])

    print(f"Generated {n} sales transactions")


def generate_monthly_metrics(conn: duckdb.DuckDBPyConnection):
    """Aggregate monthly metrics for forecasting."""

    conn.execute("""
        INSERT INTO monthly_metrics
        SELECT
            DATE_TRUNC('month', sale_date) as month_date,
            SUM(total_amount) as total_revenue,
            COUNT(*) as total_orders,
            COUNT(DISTINCT customer_id) as new_customers,
            0 as churned_customers,
            AVG(total_amount) as avg_order_value
        FROM sales
        GROUP BY DATE_TRUNC('month', sale_date)
        ORDER BY month_date
    """)

    print("Generated monthly metrics")


def seed_database():
    """Main function to seed the database."""
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))

    print("Creating tables...")
    create_tables(conn)

    print("\nGenerating data...")
    generate_products(conn)
    generate_customers(conn, n=500)
    generate_sales_reps(conn, n=25)
    generate_sales(conn, n=5000)
    generate_monthly_metrics(conn)

    # Verify data
    print("\n--- Database Summary ---")
    for table in ["customers", "products", "sales_reps", "sales", "monthly_metrics"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} rows")

    conn.close()
    print(f"\nDatabase created at: {db_path.absolute()}")


if __name__ == "__main__":
    seed_database()
