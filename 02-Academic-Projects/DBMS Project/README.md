# E-commerce Database Project

A student database project that models a small e-commerce workflow with users, items, orders, memberships, staff, and staff ratings. The SQL file includes sample records, queries, and views.

## Schema Overview

| Table | Purpose |
|-------|---------|
| `users` | Customer contact and address details |
| `item` | Product price, stock, and rating |
| `orders` | Customer purchases and totals |
| `membership` | Customer membership validity dates |
| `staff` | Staff names, departments, and salaries |
| `rating` | Customer ratings for staff |

## Project Structure

```text
DBMS Project/
├── dbms.sql                 # Database, tables, sample data, queries, and views
├── docs/
│   ├── design.md
│   ├── DBMS_Project.docx
│   └── DBMS_Project.pdf
├── dbms outputs/            # Saved output artifacts
└── README.md
```

## Requirements

- MySQL-compatible database server and command-line client

The script uses MySQL statements such as `CREATE DATABASE`, `USE`, and `SHOW TABLES`; it is not directly runnable as SQLite or PostgreSQL SQL.

## Run the SQL

From a terminal with the MySQL client installed:

```bash
mysql -u root -p < dbms.sql
```

The script creates and selects the `dbms` database, builds tables, inserts sample data, runs example queries, and creates views.

## Important Note

In `dbms.sql`, the `rating` table is created with a foreign key to `staff` before the `staff` table is created. MySQL may reject that table definition. If it does, move the `CREATE TABLE staff` statement above `CREATE TABLE rating`, then rerun the script against a clean database.

## License

No license file is included for this project.
