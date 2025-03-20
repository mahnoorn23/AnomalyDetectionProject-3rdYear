import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

# Create output folder
output_folder = "Reports"
os.makedirs(output_folder, exist_ok=True)

def generate_reports():
    """Generates daily usage reports, anomaly summaries, and cluster summaries"""

    # Load datasets
    df = pd.read_csv("Data\cleaned_data.csv")
    anomalies_df = pd.read_csv("Data/anomalies_detected.csv")
    clusters_df = pd.read_csv("Data\clustered_data.csv")

    # Convert timestamp
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df.set_index('utc_time', inplace=True)

    ### 1. Daily Water Usage Summary
    df['day'] = df.index.date
    daily_usage = df.groupby('day')['flowQuantity_delta'].sum()
    daily_usage.to_csv(f"{output_folder}/daily_usage_report.csv") # Generates report for daily total water usage

    ### 2. Asset-Wise Water Usage
    asset_usage = df.groupby('asset_type')['flowQuantity_delta'].sum()
    asset_usage.to_csv(f"{output_folder}/asset_usage_report.csv") # Generates report for total usage per asset type

    ### 3. Anomaly Summary
    anomaly_summary = anomalies_df.groupby(anomalies_df['utc_time'].str[:10])['flowQuantity_delta'].count()
    anomaly_summary.to_csv(f"{output_folder}/anomaly_summary.csv") # Generates report for daily count of detected anomalies

    ### 4. Cluster Summary
    cluster_summary = clusters_df.groupby('cluster')['flowQuantity_delta'].mean()
    cluster_summary.to_csv(f"{output_folder}/cluster_summary.csv") # Generates report for average consumption per cluster

    ### 5. Visualization: Daily Usage Trend
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, marker='o', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Total Water Usage (in Liters)")
    plt.title("Daily Water Usage Trend")
    plt.savefig(f"{output_folder}/daily_usage_trend.png") # Generates a visualization for daily water consumption trends
    plt.close()

    ### 6. Visualization: Cluster Water Consumption
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=clusters_df['cluster'], y=clusters_df['flowQuantity_delta'])
    plt.xlabel("Cluster")
    plt.ylabel("Water Comsumption (in Liters)")
    plt.title("Water Usage by Cluster")
    plt.savefig(f"{output_folder}/cluster_water_usage.png") # Generates a visualization for water consumption by cluster
    plt.close()

    ### 7. Generate the PDF Report
    generate_pdf_report(daily_usage, asset_usage, anomaly_summary, cluster_summary)

    print(f"Reports generated and saved in {output_folder}/")

def generate_pdf_report(daily_usage, asset_usage, anomaly_summary, cluster_summary):
    """Creates a PDF summary of water consumption analysis"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Water Consumption Report", ln=True, align="C")
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Daily Water Usage Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    for date, usage in daily_usage.items():
        pdf.cell(200, 7, f"{date}: {usage: .2f} Liters", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Water Usage by Asset Type", ln=True)
    pdf.set_font("Arial", "", 10)
    for asset, usage in asset_usage.items():
        pdf.cell(200, 7, f"{asset}: {usage:.2f} Liters", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Anomaly Detection Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    for date, count in anomaly_summary.items():
        pdf.cell(200, 7, f"{date}: {count} Anomalies Detected", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Cluster Water Usage", ln=True)
    pdf.set_font("Arial", "", 10)
    for cluster, avg_usage in cluster_summary.items():
        pdf.cell(200, 7, f"Cluster {cluster}: {avg_usage: .2f} Liters (Avg)", ln=True)

    pdf.output(f"{output_folder}/water_consumption_report.pdf")
    print(f"PDF report has been saved in {output_folder}/water_consumption_report.pdf")

if __name__ == "__main__":
    generate_reports()