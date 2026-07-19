Enquiry Transfer System

This repository contains a simple database management system designed for
an e-commerce platform, created as a student project by Neela Sai, Reddy
Swarith Reddy, and Kummari Shiva Charan.

📋 Overview

The system manages the following entities:

- Users (personal data, addresses, contact info)
- Items (name, price, stock, rating)
- Orders (user purchases with quantities and total amount)
- Memberships (user subscriptions with expiry dates)
- Ratings (user feedback for staff members)
- Staff (employees with department and salary information)

The accompanying `dbms.sql` file contains the schema and sample queries.

📁 Structure

```
DBMS Project/
├── .gitignore
├── README.md
├── dbms.sql               <- SQL script for tables and sample data
├── docs/
│   ├── DBMS_Project.docx  <- original report
│   ├── DBMS_Project.pdf   <- exported PDF copy
│   └── design.md          <- extracted summary of design
└── "dbms outputs"/       <- output files from execution (keep if needed)
```

🛠 Using the SQL script

Import `dbms.sql` into your favorite database server (MySQL, PostgreSQL,
SQLite, etc.) and run the provided queries to create tables and test data.

Example using SQLite:

```sh
sqlite3 ecommerce.db < dbms.sql
``` 

🤝 Contributions

Feel free to fork this project and extend the schema or provide a front-end
interface. Pull requests and issue reports are welcome.

📜 License

This project does not include a formal license—add one if you plan to share
publicly.
