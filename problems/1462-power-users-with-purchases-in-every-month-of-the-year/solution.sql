-- your query
WITH monthly_purchases AS (
    SELECT
        user_id,
        MONTH(purchase_date) AS month,
        COUNT(*) AS purchase_count
    FROM purchases
    WHERE purchase_date >= '2024-01-01'
      AND purchase_date < '2025-01-01'
    GROUP BY user_id, MONTH(purchase_date)
    HAVING COUNT(*) >= 2
)
SELECT user_id
FROM monthly_purchases
GROUP BY user_id
HAVING COUNT(*) = 12
ORDER BY user_id;