import pandas as pd
import numpy as np
import os
from datetime import datetime


class MobilePlanAnalyzer:
    """Analyzes mobile plans and recommends best options."""

    def __init__(self):
        """Initialize analyzer with default file paths."""
        base = "/workspaces/task_mobile"
        csv_dir = f"{base}/src/insights/CSV_files"
        self.customer_usage_path = (
            f"{csv_dir}/Customer_Usage_Last_12_Months_new.csv"
        )
        self.mobile_plans_path = f"{csv_dir}/Mobile_Plans_Test_Data.csv"
        self.customer_usage = None
        self.mobile_plans = None

    def load_data(self):
        """Load and preprocess customer usage and mobile plans data."""
        self.customer_usage = pd.read_csv(
            self.customer_usage_path,
            sep=';',  
            decimal=',',  
            index_col=0  
        )
        self.mobile_plans = pd.read_csv(self.mobile_plans_path)
        
        self.customer_usage = self.customer_usage.reset_index().rename(
            columns={'index': 'CustomerID'}
        )
        
        self.customer_usage = self.customer_usage.rename(columns={
            'Date': 'Month',
            'Data': 'Monthly Data Usage (GB)',
            'Minute': 'Monthly Minutes Usage',
            'SMS': 'Monthly SMS Usage'
        })
        
        self.customer_usage['Month'] = pd.to_datetime(
            self.customer_usage['Month'],
            format='%d.%m.%Y'  
        )

    def calculate_usage_statistics(self):
        """Calculate median usage statistics for each customer."""
        stats = self.customer_usage.groupby('CustomerID').agg({
            'Monthly Data Usage (GB)': 'median',
            'Monthly Minutes Usage': 'median',
            'Monthly SMS Usage': 'median'
        }).reset_index()
        return stats

    def calculate_plan_cost(self, usage, plan):
        """Calculate total cost for a given usage under a specific plan."""
        total_cost = plan['Monthly Cost ($)']
        
        data_overcharge = max(0, np.ceil(
            usage['Monthly Data Usage (GB)'] - plan['Data Limit (GB)']
        )) * plan['Data Overcharge ($/GB)']
        
        minutes_overcharge = max(
            0, usage['Monthly Minutes Usage'] - plan['Minutes Limit']
        ) * plan['Minutes Overcharge ($/min)']
        
        sms_overcharge = max(
            0, usage['Monthly SMS Usage'] - plan['SMS Limit']
        ) * plan['SMS Overcharge ($/SMS)']
        
        overcharges = {
            'Data': data_overcharge,
            'Minutes': minutes_overcharge,
            'SMS': sms_overcharge
        }
        
        return total_cost + sum(overcharges.values()), overcharges

    def find_best_plans(self, usage_stats):
        """Find the best plan for each customer with cost breakdown."""
        best_plans = []
        
        for _, customer_usage in usage_stats.iterrows():
            customer_id = customer_usage['CustomerID']
            min_cost = float('inf')
            best_plan_details = None
            
            for _, plan in self.mobile_plans.iterrows():
                total_cost, overcharges = self.calculate_plan_cost(
                    customer_usage, plan
                )
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_plan_details = {
                        'CustomerID': customer_id,
                        'Best Plan': plan['Plan Name'],
                        'Provider': plan['Provider'],
                        'Total Cost': round(total_cost, 2)
                    }
            
            best_plans.append(best_plan_details)
        
        return pd.DataFrame(best_plans)

    def analyze_plans(self):
        """Analyze plans and return best options based on median usage."""
        self.load_data()
        usage_stats = self.calculate_usage_statistics()
        return self.find_best_plans(usage_stats)


def main():
    """Run the mobile plan analysis."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = MobilePlanAnalyzer()
    comparison_df = analyzer.analyze_plans()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"plan_comparison_{timestamp}.csv")
    comparison_df.to_csv(output_file, index=False)
    
    print("\nPlan Comparison Results:")
    print(comparison_df)
    print(f"\nResults have been saved to: {output_file}")


if __name__ == "__main__":
    main() 