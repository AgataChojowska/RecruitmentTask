from db_handler import DBHandler
from datetime import datetime


if __name__ == '__main__':
    data_handler = DBHandler(db_name="sales2.db")
    if not data_handler.table_exists("historical_sales_volume"):
        data_handler.create_table_from_csv(csv_file="historical_sales_volume.csv", table_name="historical_sales_volume")

    ranked_df = data_handler.execute_query('queries/yearly_rank.sql')
    ranked_df.to_csv(f'results/yearly_rank/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)
    sales_diff_df = data_handler.execute_query('queries/sales_difference.sql')
    ranked_df.to_csv(f'results/sales_diff/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)
