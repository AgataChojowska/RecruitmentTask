WITH RankedSales AS (
    SELECT
        year,
        quarter,
        volumeSales,
        ROW_NUMBER() OVER (PARTITION BY year, quarter ORDER BY volumeSales) AS rn,
        COUNT(*) OVER (PARTITION BY year, quarter) AS total_count
    FROM historical_sales_volume
),
MedianSales AS (
    SELECT
        year,
        quarter,
        AVG(volumeSales) AS median_volumeSales
    FROM RankedSales
    WHERE rn IN (
        (total_count + 1) / 2,
        (total_count + 2) / 2
    )
    GROUP BY year, quarter
)
SELECT
    h.year,
    h.quarter,
    h.product,
    h.volumeSales,
    m.median_volumeSales,
    ROUND(((h.volumeSales - m.median_volumeSales) / m.median_volumeSales) * 100, 2) AS percentage_difference
FROM historical_sales_volume h
JOIN MedianSales m
    ON h.year = m.year
    AND h.quarter = m.quarter
ORDER BY h.year, h.quarter, h.product;