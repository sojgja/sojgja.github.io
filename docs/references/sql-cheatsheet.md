---
id: sql-cheatsheet
title: SQL Cheatsheet
sidebar_label: 💾 SQL Cheatsheet
sidebar_position: 4
---

# SQL Cheatsheet

## SELECT

```sql
SELECT col1, col2 FROM table WHERE condition ORDER BY col1 LIMIT 10;
```

## JOIN

```sql
SELECT * FROM a JOIN b ON a.id = b.a_id;
SELECT * FROM a LEFT JOIN b ON a.id = b.a_id;
```

## GROUP BY

```sql
SELECT category, COUNT(*) FROM products GROUP BY category;
SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 100;
```

## INSERT / UPDATE / DELETE

```sql
INSERT INTO table (col1, col2) VALUES (val1, val2);
UPDATE table SET col1 = val1 WHERE condition;
DELETE FROM table WHERE condition;
```

## INDEX

```sql
CREATE INDEX idx_name ON table (col);
CREATE UNIQUE INDEX idx_unique ON table (col);
```

## Transaction

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```
