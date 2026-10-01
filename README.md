# InsightIQ — E-Commerce Customer Intelligence Platform

InsightIQ is an end-to-end customer analytics platform that transforms e-commerce transaction data into customer, revenue, segmentation, and churn insights.

The application is designed to work with a company's own transaction CSV by automatically identifying important fields such as customer, transaction, date, and revenue information from the uploaded data.

**[Live Demo](https://ecommercecustomeranalytics-5qphtkcnheakbpjhb7hktt.streamlit.app/)** · **[Source Code](https://github.com/Kaustubh7002-code/Ecommerce_Customer_Analytics)**

## Overview
E-commerce businesses generate large amounts of transaction data, but raw transaction records do not directly provide useful information about customer behavior.

InsightIQ processes transaction data and provides a single dashboard for:

- Customer analytics
- Revenue analysis
- Purchasing behavior
- RFM segmentation
- Churn analysis
- Machine-learning-based churn prediction
- Customer-level risk analysis
- Downloadable reports

## Key Features

### Customer Analytics

Provides an overview of customer activity, including:

- Total customers
- Orders and transactions
- Average order value
- Customer spending
- Repeat purchasing behavior
- High-value customers

### Revenue Analysis

Analyzes transaction-level revenue to identify:

- Overall revenue
- Revenue trends
- Average order value
- Customer contribution to revenue
- Product/category performance when available

### RFM Segmentation

Customers are analyzed using:

- Recency — how recently a customer purchased
- Frequency — how often a customer purchased
- Monetary — how much a customer spent

RFM scores are used to understand different customer segments and purchasing behavior.

### Churn Analysis

The platform identifies customers based on their recent purchasing activity and analyzes potential churn risk.

Users can configure the churn observation window based on their dataset.

### Churn Prediction

Machine learning is used to estimate the probability of customer churn using historical customer behavior.

The model uses behavioral features such as:

- Recency
- Frequency
- Monetary value
- Total items purchased
- Average order value
- Customer lifetime

Customers are assigned risk levels based on their predicted churn probability.

## Automatic Data Understanding

A major feature of InsightIQ is its ability to work with different e-commerce transaction datasets.

Instead of depending on fixed column names, the application analyzes the uploaded dataset and attempts to identify:

- Customer / buyer field
- Transaction / order field
- Transaction date
- Revenue / purchase amount
- Quantity, when available
- Product or category, when available

This allows the platform to handle transaction datasets with different column names and structures.

The application also reports the detected fields and confidence information before analysis.

## Project Workflow

```text
Transaction CSV
       |
       v
Automatic Data Understanding
       |
       v
Schema Inference
       |
       v
Data Cleaning & Standardization
       |
       v
Customer & Revenue Analytics
       |
       v
RFM Segmentation
       |
       v
Churn Analysis
       |
       v
Machine Learning
       |
       v
Churn Probability & Risk Levels
       |
       v
Interactive Dashboard
