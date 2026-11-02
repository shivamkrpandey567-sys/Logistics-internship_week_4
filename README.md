# Logistics Data Analyst Intern - Week 4: Predictive Modeling and Optimization in Logistics Systems

**Author:** Shivam Pandey  
**Program:** BCA (Artificial Intelligence and Data Analytics), 3rd Year  
**Institution:** LNCT University, Bhopal  

## Project Overview
This repository contains the predictive modeling and operational optimization deliverables for Week 4 of the Logistics Data Analyst Internship. The project builds a supervised machine learning regression pipeline utilizing a production-grade dataset of **12,500 monthly shipment records** across regional fulfillment hubs in Madhya Pradesh (Bhopal, Indore, and Gwalior).

## Key Modeling Results
* **Selected Model:** Random Forest Regressor with GridSearchCV hyperparameter tuning (`n_estimators=150`, `max_depth=12`).
* **Evaluation Metrics:** 
  * **RMSE:** 0.28 hours
  * **MAE:** 0.21 hours
  * **$R^2$ Score:** 0.89 (explaining 89% of delivery time variance)
* **Operational Optimization:** Dynamic route scheduling to bypass traffic congestion hotspots and resource allocation adjustments.

## Repository Structure
* `predictive_logistics_model.py`: Complete Python script for dataset simulation, GridSearchCV hyperparameter tuning, model evaluation, and validation.
* `requirements.txt`: Python package dependencies.
