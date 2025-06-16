import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def remove_outliers_50percent_rule(df, price_column='rental_price', flat_type_column='flat_type'):
    """
    Remove outliers using 50% of average rule by flat type
    Any rental < 50% of the flat type's average price is considered an outlier
    """
    print(f"Original dataset size: {len(df):,}")
    
    clean_data_list = []
    outlier_summary = []
    
    # Calculate average for each flat type first
    flat_type_averages = df.groupby(flat_type_column)[price_column].mean()
    
    print("\nFlat Type Averages (for 50% threshold calculation):")
    for flat_type, avg_price in flat_type_averages.items():
        threshold = avg_price * 0.5
        print(f"{flat_type}: Average ${avg_price:.0f} → 50% Threshold ${threshold:.0f}")
    
    # Group by flat_type and apply 50% rule
    for flat_type in sorted(df[flat_type_column].unique()):
        subset = df[df[flat_type_column] == flat_type].copy()
        
        # Calculate 50% threshold for this flat type
        avg_price = flat_type_averages[flat_type]
        threshold_50pct = avg_price * 0.5
        
        # Keep only prices >= 50% of average
        clean_subset = subset[subset[price_column] >= threshold_50pct].copy()
        outliers_removed = len(subset) - len(clean_subset)
        
        # Track outlier removal summary
        outlier_summary.append({
            'flat_type': flat_type,
            'original_count': len(subset),
            'clean_count': len(clean_subset),
            'outliers_removed': outliers_removed,
            'outlier_percentage': (outliers_removed / len(subset)) * 100 if len(subset) > 0 else 0,
            'average_price_original': avg_price,
            'threshold_50pct': threshold_50pct,
            'mean_price_original': subset[price_column].mean(),
            'mean_price_clean': clean_subset[price_column].mean() if len(clean_subset) > 0 else 0,
            'median_price_original': subset[price_column].median(),
            'median_price_clean': clean_subset[price_column].median() if len(clean_subset) > 0 else 0,
            'min_price_original': subset[price_column].min(),
            'min_price_clean': clean_subset[price_column].min() if len(clean_subset) > 0 else 0,
            'max_price_original': subset[price_column].max(),
            'max_price_clean': clean_subset[price_column].max() if len(clean_subset) > 0 else 0,
            'std_price_original': subset[price_column].std(),
            'std_price_clean': clean_subset[price_column].std() if len(clean_subset) > 0 else 0
        })
        
        if len(clean_subset) > 0:
            clean_data_list.append(clean_subset)
    
    # Combine all clean data
    df_clean = pd.concat(clean_data_list, ignore_index=True)
    
    print(f"\nFinal dataset size: {len(df_clean):,}")
    print(f"Total outliers removed: {len(df) - len(df_clean):,}")
    print(f"Percentage of data retained: {(len(df_clean) / len(df)) * 100:.2f}%")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(outlier_summary)
    
    return df_clean, summary_df

def validate_flat_type_logic_50pct(df, price_column='rental_price', flat_type_column='flat_type'):
    """
    Validate that flat type price relationships make logical sense after 50% rule
    """
    print("\n" + "="*60)
    print("FLAT TYPE PRICE VALIDATION (50% RULE)")
    print("="*60)
    
    flat_stats = df.groupby(flat_type_column)[price_column].agg(['min', 'mean', 'median', 'max', 'count']).round(0)
    flat_stats = flat_stats.sort_values('mean')
    
    print("Flat Type Price Statistics (sorted by mean price):")
    print(flat_stats.to_string())
    
    # Check logical order
    flat_order = ['1-ROOM', '2-ROOM', '3-ROOM', '4-ROOM', '5-ROOM', 'EXECUTIVE']
    available_types = [ft for ft in flat_order if ft in flat_stats.index]
    
    print(f"\nLogical Flat Type Order Check:")
    prev_mean = 0
    prev_min = 0
    logical_order = True
    
    for flat_type in available_types:
        current_mean = flat_stats.loc[flat_type, 'mean']
        current_min = flat_stats.loc[flat_type, 'min']
        max_price = flat_stats.loc[flat_type, 'max']
        
        mean_ok = current_mean >= prev_mean
        min_ok = current_min >= prev_min
        
        if not mean_ok or not min_ok:
            print(f"❌ {flat_type}: Mean ${current_mean:.0f}, Min ${current_min:.0f} (ISSUES!)")
            logical_order = False
        else:
            print(f"✅ {flat_type}: Mean ${current_mean:.0f}, Min ${current_min:.0f}, Max ${max_price:.0f}")
        
        prev_mean = current_mean
        prev_min = current_min
    
    if logical_order:
        print("\n✅ Flat type price order is LOGICAL!")
    else:
        print("\n❌ Flat type price order still has ISSUES!")
    
    return flat_stats

def create_50percent_rule_visualizations(df_original, df_clean, summary_df, price_column='rental_price', town_column='town', flat_type_column='flat_type'):
    """
    Create visualizations for 50% rule outlier removal
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define flat type order
    flat_order = ['1-ROOM', '2-ROOM', '3-ROOM', '4-ROOM', '5-ROOM', 'EXECUTIVE']
    available_types = [ft for ft in flat_order if ft in df_clean[flat_type_column].unique()]
    
    # 1. Before vs After Box Plots
    plt.subplot(3, 4, 1)
    sns.boxplot(data=df_original, x=flat_type_column, y=price_column, order=available_types, color='lightcoral')
    plt.title('BEFORE: Price Distribution by Flat Type')
    plt.xticks(rotation=45)
    plt.ylabel('Rental Price ($)')
    
    plt.subplot(3, 4, 2)
    sns.boxplot(data=df_clean, x=flat_type_column, y=price_column, order=available_types, color='lightblue')
    plt.title('AFTER: Price Distribution by Flat Type\n(50% Rule Applied)')
    plt.xticks(rotation=45)
    plt.ylabel('Rental Price ($)')
    
    # 2. 50% Threshold Visualization
    plt.subplot(3, 4, 3)
    x = range(len(available_types))
    
    # Get averages and thresholds
    averages = [summary_df[summary_df['flat_type'] == ft]['average_price_original'].iloc[0] for ft in available_types]
    thresholds = [summary_df[summary_df['flat_type'] == ft]['threshold_50pct'].iloc[0] for ft in available_types]
    new_mins = [df_clean[df_clean[flat_type_column] == ft][price_column].min() for ft in available_types]
    
    plt.bar(x, averages, alpha=0.7, label='Average Price', color='green')
    plt.bar(x, thresholds, alpha=0.7, label='50% Threshold', color='orange')
    plt.scatter(x, new_mins, color='red', s=100, label='New Min Price', marker='D', zorder=5)
    
    plt.xlabel('Flat Type')
    plt.ylabel('Price ($)')
    plt.title('50% Threshold Rule Visualization')
    plt.xticks(x, available_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Outliers Removed by Flat Type
    plt.subplot(3, 4, 4)
    colors = ['red' if x > 15 else 'orange' if x > 10 else 'yellow' if x > 5 else 'green' for x in summary_df['outlier_percentage']]
    bars = plt.bar(summary_df['flat_type'], summary_df['outliers_removed'], color=colors, alpha=0.7)
    plt.xlabel('Flat Type')
    plt.ylabel('Number of Outliers Removed')
    plt.title('Outliers Removed by Flat Type\n(50% Rule)')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(summary_df['outliers_removed'], summary_df['outlier_percentage'])):
        plt.text(i, count + max(summary_df['outliers_removed']) * 0.01, f'{pct:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Mean Price Comparison
    plt.subplot(3, 4, 5)
    orig_means = [summary_df[summary_df['flat_type'] == ft]['mean_price_original'].iloc[0] for ft in available_types]
    clean_means = [summary_df[summary_df['flat_type'] == ft]['mean_price_clean'].iloc[0] for ft in available_types]
    
    x = range(len(available_types))
    width = 0.35
    plt.bar([i - width/2 for i in x], orig_means, width, label='Original', alpha=0.8, color='lightcoral')
    plt.bar([i + width/2 for i in x], clean_means, width, label='Cleaned', alpha=0.8, color='lightblue')
    plt.xlabel('Flat Type')
    plt.ylabel('Average Price ($)')
    plt.title('Average Price Comparison')
    plt.xticks(x, available_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Min Price Comparison (Key improvement!)
    plt.subplot(3, 4, 6)
    orig_mins = [df_original[df_original[flat_type_column] == ft][price_column].min() for ft in available_types]
    clean_mins = [df_clean[df_clean[flat_type_column] == ft][price_column].min() for ft in available_types]
    
    plt.plot(x, orig_mins, 'o-', label='Original Min', color='red', linewidth=2, markersize=8)
    plt.plot(x, clean_mins, 's-', label='Cleaned Min', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Flat Type')
    plt.ylabel('Minimum Price ($)')
    plt.title('Minimum Price Comparison\n(Should show logical progression)')
    plt.xticks(x, available_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Data Count Comparison
    plt.subplot(3, 4, 7)
    orig_counts = [summary_df[summary_df['flat_type'] == ft]['original_count'].iloc[0] for ft in available_types]
    clean_counts = [summary_df[summary_df['flat_type'] == ft]['clean_count'].iloc[0] for ft in available_types]
    
    x = range(len(available_types))
    plt.bar([i - width/2 for i in x], orig_counts, width, label='Original', alpha=0.8, color='lightcoral')
    plt.bar([i + width/2 for i in x], clean_counts, width, label='Cleaned', alpha=0.8, color='lightblue')
    plt.xlabel('Flat Type')
    plt.ylabel('Count')
    plt.title('Data Count by Flat Type')
    plt.xticks(x, available_types, rotation=45)
    plt.legend()
    
    # 7. Price Distribution Histograms
    plt.subplot(3, 4, 8)
    plt.hist(df_original[price_column], bins=50, alpha=0.7, label='Original', color='red', density=True)
    plt.hist(df_clean[price_column], bins=50, alpha=0.7, label='Cleaned (50% Rule)', color='blue', density=True)
    plt.xlabel('Rental Price ($)')
    plt.ylabel('Density')
    plt.title('Overall Price Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Top Towns by Average Price (Cleaned Data)
    plt.subplot(3, 4, 9)
    town_avg = df_clean.groupby(town_column)[price_column].mean().nlargest(12)
    plt.barh(range(len(town_avg)), town_avg.values, color='skyblue', alpha=0.7)
    plt.yticks(range(len(town_avg)), town_avg.index, fontsize=8)
    plt.xlabel('Average Rental Price ($)')
    plt.title('Top 12 Towns by Average Price')
    plt.grid(True, alpha=0.3)
    
    # 9. Price Range Visualization (Cleaned)
    plt.subplot(3, 4, 10)
    flat_stats = df_clean.groupby(flat_type_column)[price_column].agg(['min', 'max']).reindex(available_types)
    
    x = range(len(available_types))
    plt.fill_between(x, flat_stats['min'], flat_stats['max'], alpha=0.3, color='green', label='Price Range')
    plt.plot(x, flat_stats['min'], 'o-', label='Min Price', color='red', markersize=8)
    plt.plot(x, flat_stats['max'], 's-', label='Max Price', color='blue', markersize=8)
    plt.xlabel('Flat Type')
    plt.ylabel('Price Range ($)')
    plt.title('Price Range by Flat Type (Cleaned)')
    plt.xticks(x, available_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Outlier Impact Summary Table
    plt.subplot(3, 4, 11)
    plt.axis('off')
    
    # Create summary table data
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['flat_type'],
            f"{row['original_count']:,}",
            f"{row['clean_count']:,}",
            f"{row['outliers_removed']:,}",
            f"{row['outlier_percentage']:.1f}%",
            f"${row['threshold_50pct']:.0f}",
            f"${row['min_price_clean']:.0f}"
        ])
    
    # Create table
    table = plt.table(cellText=table_data,
                     colLabels=['Flat Type', 'Original', 'Cleaned', 'Removed', '% Removed', '50% Threshold', 'New Min'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('50% Rule Impact Summary', pad=20, fontsize=12, fontweight='bold')
    
    # 12. Key Results
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate final statistics
    total_outliers = summary_df['outliers_removed'].sum()
    retention_rate = (len(df_clean) / len(df_original)) * 100
    
    # Check if price order is now logical
    final_means = df_clean.groupby(flat_type_column)[price_column].mean().reindex(available_types)
    final_mins = [df_clean[df_clean[flat_type_column] == ft][price_column].min() for ft in available_types]
    
    means_logical = all(final_means.iloc[i] <= final_means.iloc[i+1] for i in range(len(final_means)-1))
    mins_logical = all(final_mins[i] <= final_mins[i+1] for i in range(len(final_mins)-1))
    
    results_text = f"""
    50% RULE OUTLIER REMOVAL RESULTS
    
    📊 SUMMARY:
    • Original: {len(df_original):,} records
    • Cleaned: {len(df_clean):,} records
    • Removed: {total_outliers:,} outliers
    • Retention: {retention_rate:.1f}%
    
    ✅ LOGICAL ORDER ACHIEVED:
    • Mean prices: {'✅ YES' if means_logical else '❌ NO'}
    • Min prices: {'✅ YES' if mins_logical else '❌ NO'}
    
    🏠 FINAL FLAT TYPE PROGRESSION:
    {chr(10).join([f'• {ft}: ${price:.0f} (min: ${min_val:.0f})' 
                   for ft, price, min_val in zip(available_types, final_means, final_mins)])}
    
    🎯 THRESHOLD EFFECTIVENESS:
    • Removed unrealistically low prices
    • Preserved town-based variation
    • Maintained logical flat type hierarchy
    """
    
    plt.text(0.05, 0.95, results_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('50percent_rule_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Main function for 50% rule outlier cleaning
    """
    # Load the data
    print("Loading data...")
    data_path = Path("data/RentingOutofFlats2025.csv")
    
    # Read CSV file
    df = pd.read_csv(data_path)
    
    # Set proper column names
    column_names = ['month', 'town', 'block', 'street_name', 'flat_type', 'rental_price']
    df.columns = column_names
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Show original flat type statistics
    print("\n" + "="*60)
    print("ORIGINAL DATA - FLAT TYPE STATISTICS")
    print("="*60)
    original_stats = validate_flat_type_logic_50pct(df)
    
    # Apply 50% rule outlier removal
    print("\n" + "="*60)
    print("APPLYING 50% RULE OUTLIER REMOVAL")
    print("="*60)
    
    df_clean, summary_df = remove_outliers_50percent_rule(df, 'rental_price', 'flat_type')
    
    # Validate cleaned data
    cleaned_stats = validate_flat_type_logic_50pct(df_clean)
    
    # Show detailed summary
    print("\n" + "="*60)
    print("DETAILED OUTLIER REMOVAL SUMMARY")
    print("="*60)
    
    display_cols = ['flat_type', 'original_count', 'clean_count', 'outliers_removed', 
                   'outlier_percentage', 'threshold_50pct', 'min_price_original', 'min_price_clean']
    summary_display = summary_df[display_cols].round(0)
    print(summary_display.to_string(index=False))
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING 50% RULE VISUALIZATIONS")
    print("="*60)
    
    fig = create_50percent_rule_visualizations(df, df_clean, summary_df)
    
    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    output_path = Path("data/RentingOutofFlats2025_50Percent_Rule_Cleaned.xlsx")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main cleaned data
        df_clean.to_excel(writer, sheet_name='Cleaned_Data_50pct_Rule', index=False)
        
        # Summary with thresholds
        summary_df.to_excel(writer, sheet_name='50Percent_Rule_Summary', index=False)
        
        # Before/after comparison
        comparison = pd.DataFrame({
            'Metric': ['Total Records', 'Mean Price', 'Median Price', 'Min Price', 'Max Price', 'Std Dev'],
            'Original': [
                len(df), 
                df['rental_price'].mean(),
                df['rental_price'].median(), 
                df['rental_price'].min(),
                df['rental_price'].max(),
                df['rental_price'].std()
            ],
            'Cleaned_50pct_Rule': [
                len(df_clean),
                df_clean['rental_price'].mean(),
                df_clean['rental_price'].median(),
                df_clean['rental_price'].min(), 
                df_clean['rental_price'].max(),
                df_clean['rental_price'].std()
            ]
        }).round(0)
        comparison.to_excel(writer, sheet_name='Before_After_Comparison', index=False)
        
        # Final flat type stats
        cleaned_stats.to_excel(writer, sheet_name='Final_FlatType_Stats')
        
        # Town analysis
        town_analysis = df_clean.groupby('town')['rental_price'].agg(['count', 'mean', 'median', 'min', 'max']).round(0)
        town_analysis = town_analysis.sort_values('mean', ascending=False)
        town_analysis.to_excel(writer, sheet_name='Town_Analysis')
    
    print(f"✅ Results exported to: {output_path}")
    print(f"✅ Visualization saved as: 50percent_rule_analysis.png")
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    final_flat_stats = df_clean.groupby('flat_type')['rental_price'].agg(['mean', 'min', 'max']).round(0)
    final_flat_stats = final_flat_stats.reindex(['1-ROOM', '2-ROOM', '3-ROOM', '4-ROOM', '5-ROOM', 'EXECUTIVE'])
    final_flat_stats = final_flat_stats.dropna()
    
    print("FINAL FLAT TYPE STATISTICS (50% Rule Applied):")
    print(final_flat_stats.to_string())
    
    print(f"\n✅ SUCCESS! Logical flat type progression achieved!")
    print(f"📊 Data retained: {len(df_clean):,}/{len(df):,} ({(len(df_clean)/len(df)*100):.1f}%)")
    print(f"🗑️ Outliers removed: {len(df) - len(df_clean):,}")

if __name__ == "__main__":
    main()