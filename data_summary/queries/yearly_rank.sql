WITH CategorizedSales AS (
    SELECT
        h.year,
        CASE
            WHEN h.product IN ('P1', 'P2', 'P3') THEN 'C1'
            WHEN h.product IN ('P4', 'P5', 'P6') THEN 'C2'
            ELSE 'Unknown'
        END AS category,
        h.product,
        SUM(h.volumeSales) AS total_sales
    FROM historical_sales_volume h
    GROUP BY h.year, category, h.product
),
RankedSales AS (
    SELECT
        year,
        category,
        product,
        total_sales,
        RANK() OVER (PARTITION BY year, category ORDER BY total_sales DESC) AS rank
    FROM CategorizedSales
)
SELECT *
FROM RankedSales
ORDER BY year, category, rank;