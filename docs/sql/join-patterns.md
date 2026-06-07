---
id: join-patterns
title: SQL JOIN Patterns
sidebar_label: JOIN Patterns
sidebar_position: 2
description: Essential SQL JOIN patterns — INNER, LEFT, CROSS, anti-join, semi-join, lateral joins for trading data analysis and reporting.
keywords: [sql, join, inner-join, left-join, cross-join, lateral, window-function]
---

# SQL JOIN Patterns

Production-grade JOIN patterns for financial and trading data.

## INNER JOIN — Match Only

```sql
-- Match orders with their fills
SELECT o.order_id, o.symbol, f.fill_price, f.fill_qty
FROM orders o
INNER JOIN fills f ON o.order_id = f.order_id
WHERE o.created_at >= SYSDATE - 1;
```

## LEFT JOIN — Keep All Left

```sql
-- All orders, whether filled or not
SELECT o.order_id, NVL(f.fill_qty, 0) AS filled_qty
FROM orders o
LEFT JOIN fills f ON o.order_id = f.order_id;
```

## Anti-Join (NOT EXISTS)

```sql
-- Orders with NO fills yet (unfilled orders)
SELECT o.* FROM orders o
WHERE NOT EXISTS (
    SELECT 1 FROM fills f WHERE f.order_id = o.order_id
);
```

## Semi-Join (EXISTS)

```sql
-- Symbols that had at least one trade today
SELECT DISTINCT s.symbol
FROM symbols s
WHERE EXISTS (
    SELECT 1 FROM trades t
    WHERE t.symbol = s.symbol AND t.trade_date = TRUNC(SYSDATE)
);
```

## CROSS JOIN — Cartesian

```sql
-- Generate all symbol × date combinations for a calendar report
SELECT s.symbol, d.trade_date
FROM symbols s
CROSS JOIN trading_dates d
WHERE d.trade_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-31';
```

## Window Function with JOIN

```sql
-- Running total of fills per order
SELECT o.order_id, f.fill_price, f.fill_qty,
       SUM(f.fill_qty) OVER (PARTITION BY o.order_id ORDER BY f.fill_time) AS cumulative_qty
FROM orders o
JOIN fills f ON o.order_id = f.order_id;
```
