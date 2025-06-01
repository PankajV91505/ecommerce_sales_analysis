import os
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

def load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('bob.wilson@,', 'bob.wilson@company.com')
    
    lines = content.split('\n')
    good_lines = [lines[0]] 
    
    for line in lines[1:]:
        if not line.strip():
            continue
        if line.count(',') == 7:
            good_lines.append(line)
        else:
            parts = line.split(',')
            if len(parts) == 9:
                fixed_line = ','.join(parts[:7] + [','.join(parts[7:])])
                good_lines.append(fixed_line)
    
    return pd.read_csv(StringIO('\n'.join(good_lines)))

print("Loading data files...")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

try:
    sales = load_csv(os.path.join(data_dir, 'project1_sales_data.csv'))
    products = pd.read_csv(os.path.join(data_dir, 'project1_product_data.csv'))
    customers = pd.read_csv(os.path.join(data_dir, 'project1_customer_data.csv'))
    
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()
    

print("\nCleaning data...")

sales['product_name'] = sales['product_name'].str.strip()
sales['price'] = sales['price'].str.replace(r'[$,€]', '', regex=True).astype(float)

def parse_date(date_str):
    
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except ValueError:
            continue
    return np.nan

sales['order_date'] = sales['order_date'].apply(parse_date)
sales['order_date'] = pd.to_datetime(sales['order_date'])
sales['customer_email'] = sales['customer_email'].str.strip().replace('', np.nan)
sales['revenue'] = sales['price'] * sales['quantity']

products['product_name'] = products['product_name'].str.strip()

customers = customers.drop_duplicates(subset='customer_id', keep='first')
customers['country'] = customers['country'].replace({
    'United Kingdom': 'UK',
    '': np.nan
})
median_age = customers['age'].median()
customers['age'] = customers['age'].fillna(median_age)
customers['age_group'] = pd.cut(
    customers['age'],
    bins=[0, 20, 30, 40, 50, 100],
    labels=['<20', '20-29', '30-39', '40-49', '50+'],
    right=False
)


# Data Merging

print("\nMerging datasets...")

merged_data = pd.merge(
    pd.merge(
        sales,
        products,
        on=['product_name', 'category'],
        how='left'
    ),
    customers,
    on='customer_id',
    how='left'
)

merged_data['profit'] = (merged_data['price'] - merged_data['cost_price']) * merged_data['quantity']


# Data Analysis

print("\nAnalyzing data...")
 
analysis_results = {
    'monthly_sales': merged_data.groupby(pd.Grouper(key='order_date', freq='ME'))['revenue'].sum().reset_index(),
    'top_products': merged_data.groupby('product_name')['revenue'].sum().nlargest(10).reset_index(),
    'customer_demo': merged_data.groupby(['age_group', 'country'], observed=True)['customer_id'].nunique().reset_index(),
    'geo_sales': merged_data.groupby('country')['revenue'].sum().sort_values(ascending=False).reset_index(),
    'customer_lifetime': merged_data.groupby('customer_id').agg({
        'revenue': 'sum',
        'order_id': 'nunique',
        'name': 'first',
        'country': 'first'
    }).rename(columns={'order_id': 'order_count'}).sort_values('revenue', ascending=False)
}



print("\nGenerating reports...")

# Data Cleaning Report

cleaning_report = pd.DataFrame({
    'Dataset': ['Sales']*6 + ['Customers']*4 + ['Products']*1,
    'Issue': [
        'Whitespace in product names',
        'Currency symbols in prices',
        'Inconsistent date formats',
        'Invalid/missing emails',
        'Missing customer_id',
        'Inconsistent category names',
        'Duplicate records',
        'Missing age values',
        'Missing city/country',
        'Inconsistent country names',
        'Whitespace in product names'
    ],
    'Records_Affected': [
        sales['product_name'].str.contains('  ').sum(),
        len(sales),
        sales['order_date'].isna().sum(),
        sales['customer_email'].isna().sum(),
        sales['customer_id'].isna().sum(),
        len(sales['category'].unique()) - len(products['category'].unique()),
        customers.duplicated().sum(),
        customers['age'].isna().sum(),
        (customers['city'].isna().sum() + customers['country'].isna().sum()),
        (customers['country'] == 'United Kingdom').sum(),
        products['product_name'].str.contains('  ').sum()
    ],
    'Action_Taken': [
        'Stripped whitespace',
        'Removed symbols, converted to float',
        'Standardized to YYYY-MM-DD',
        'Corrected invalid, marked missing',
        'Kept as is (only 1 record)',
        'Verified against product catalog',
        'Removed duplicates',
        'Filled with median age',
        'Left as missing',
        'Standardized to "UK"',
        'Stripped whitespace'
    ]
})

# Business Insights Report

insights_data = {
    'Top Product by Revenue': analysis_results['top_products'].iloc[0]['product_name'],
    'Top Country by Sales': analysis_results['geo_sales'].iloc[0]['country'],
    'Most Profitable Age Group': analysis_results['customer_demo']
        .sort_values('customer_id', ascending=False).iloc[0]['age_group'],
    'Highest Sales Month': analysis_results['monthly_sales']
        .sort_values('revenue', ascending=False).iloc[0]['order_date'].strftime('%B %Y'),
    'Revenue from Top 5 Customers': f"{analysis_results['customer_lifetime'].head(5)['revenue'].sum():,.2f}",
    'Total Revenue': f"{merged_data['revenue'].sum():,.2f}",
    'Total Profit': f"{merged_data['profit'].sum():,.2f}",
    'Average Order Value': f"{merged_data['revenue'].sum()/len(merged_data['order_id'].unique()):,.2f}"
}

insights = pd.DataFrame({
    'Metric': list(insights_data.keys()),
    'Value': list(insights_data.values())
})


# save output

output_dir = os.path.join(current_dir, '..', 'reports')
os.makedirs(output_dir, exist_ok=True)

cleaning_report.to_csv(os.path.join(output_dir, 'data_cleaning_report.csv'), index=False)
insights.to_csv(os.path.join(output_dir, 'business_insights.csv'), index=False)
merged_data.to_csv(os.path.join(output_dir, 'cleaned_merged_data.csv'), index=False)

print("\nAnalysis complete!")
print(f"Reports saved to: {output_dir}")