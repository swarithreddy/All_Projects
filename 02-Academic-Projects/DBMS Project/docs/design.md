System Design Summary

Project Name: Enquiry Transfer System

Entities and tables

- Users: `user_id`, `user_name`, `phone_no`, `email`, `address`
- Items: `item_id`, `item_name`, `price`, `stock`, `rating`
- Orders: `order_id`, `user_id` (FK), `bill_date`, `item_id` (FK),
  `quantity_sold`, `total_amount`
- Memberships: `mem_id`, `user_id` (FK), `valid_till` (expiry date)
- Staff: `staff_id`, `staff_name`, `department`, `salary`, `rating`
- Ratings: `cus_id` (FK to Users), `to_staff` (FK to Staff), `rating,
  `timestamp`

Core functionality

- User registration/login with personal details stored in the users table.
- Item inventory tracking, including price and rating information.
- Order processing: orders recorded, inventory updated, totals calculated.
- Membership applications with expiry tracking.
- Customer ratings of staff for performance evaluation.
- Staff data management including department and salary.

Notes

SQL definitions and example queries are available in `dbms.sql`.
Diagram and full requirements were provided in the original report
(see the Word/PDF in this `docs` folder).
