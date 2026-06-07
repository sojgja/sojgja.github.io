---
id: oracle-minus
title: Oracle MINUS & Set Operations
sidebar_label: Oracle MINUS
sidebar_position: 1
description: Oracle SQL MINUS, INTERSECT, UNION ALL patterns with practical examples for data comparison and reconciliation in trading databases.
keywords: [oracle, sql, minus, intersect, union, set-operations, data-comparison]
---

# Oracle MINUS & Set Operations

Oracle's `MINUS` operator compares result sets and returns rows from the first query that are NOT in the second query.

## MINUS — Find Missing Rows

```sql
-- Find trades in staging that haven't been synced to production
SELECT trade_id, symbol, quantity, price
FROM staging_trades
MINUS
SELECT trade_id, symbol, quantity, price
FROM production_trades;
```

## INTERSECT — Find Common Rows

```sql
-- Find trades present in BOTH staging and production
SELECT trade_id FROM staging_trades
INTERSECT
SELECT trade_id FROM production_trades;
```

## UNION ALL — Merge Data (No Dedup)

```sql
-- Combine today's trades with yesterday's for a report
SELECT * FROM trades_20260107
UNION ALL
SELECT * FROM trades_20260106;
```

:::tip Performance
`UNION ALL` is faster than `UNION` because it skips the sort/unique step. Always prefer `UNION ALL` unless you explicitly need deduplication.
:::

## Practical: Account Reconciliation

```sql
-- Find accounts with balance discrepancies
SELECT account_id, SUM(amount) AS balance
FROM journal_entries GROUP BY account_id
MINUS
SELECT account_id, balance
FROM account_balances;
```
