import pandas as pd
from model_predictor import PlanPredictor, predict_single_plan, batch_predict_plans

csv_path = '/workspaces/task_mobile/src/insights/CSV_files/Customer_Usage_Last_12_Months.csv'
customer_usage = pd.read_csv(csv_path)

customer_medians = customer_usage.groupby('CustomerID').agg({
    'Monthly Data Usage (GB)': 'median',
    'Monthly Minutes Usage': 'median',
    'Monthly SMS Usage': 'median'
}).round(2)

print("=== Customer Median Usage Analysis ===")

try:
    first_customer = customer_medians.iloc[0]
    result = predict_single_plan(
        data=first_customer['Monthly Data Usage (GB)'],
        minutes=first_customer['Monthly Minutes Usage'],
        sms=first_customer['Monthly SMS Usage']
    )
    print(f"\nRecommended plan for customer {customer_medians.index[0]}:")
    print(f"  Median Usage: Data={first_customer['Monthly Data Usage (GB)']}GB, "
          f"Minutes={first_customer['Monthly Minutes Usage']}, "
          f"SMS={first_customer['Monthly SMS Usage']}")
    print(f"  Recommended Plan: {result}")

except Exception as e:
    print(f"Error processing single prediction: {e}")

try:
    usage_data = customer_medians[['Monthly Data Usage (GB)', 
                                 'Monthly Minutes Usage', 
                                 'Monthly SMS Usage']].values
    results = batch_predict_plans(usage_data)

    print("\n=== All Customers Analysis (Based on 12-Month Medians) ===")
    for i, (customer_id, plan) in enumerate(zip(customer_medians.index, results)):
        usage = usage_data[i]
        print(f"\nCustomer {customer_id}:")
        print(f"  Median Usage: Data={usage[0]}GB, Minutes={usage[1]}, SMS={usage[2]}")
        print(f"  Recommended Plan: {plan}")

except Exception as e:
    print(f"Error processing batch predictions: {e}")

predictor = PlanPredictor()
model_info = predictor.get_model_info()
print(f"\nModel information: {model_info}")