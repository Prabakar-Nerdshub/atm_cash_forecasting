"""
URL configuration for atm_cash_forecasting project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# atm_forecasting_api/urls.py
from django.urls import path
from atm_forecasting_api import views
from .views import (XGBoostGraphData, ProphetGraphData,
                    LSTMGraphData, SarimaxGraphData, get_forecast_data)

urlpatterns = [
    path('data/', get_forecast_data, name='get_forecast_data'),
    path('', views.index, name='index'),  # Home page
    path('redirect_model/', views.redirect_model, name='redirect_model'),  # Redirect model
    path('xgboost_analysis/', views.xgboost_view, name='xgboost_analysis'),  # XGBoost analysis
    path('prophet_analysis/', views.prophet_view, name='prophet_analysis'),  # Prophet analysis
    path('lstm_analysis/', views.lstm_view, name='lstm_analysis'),  # LSTM analysis
    path('sarimax_analysis/', views.sarimax_view, name='sarimax_analysis'),  # SARIMAX analysis
    path('xgboost-graph-data/', XGBoostGraphData.as_view(), name='xgboost-graph-data'),
    path('prophet/data/', ProphetGraphData.as_view(), name='prophet-graph-data'),
    path('lstm/data/', LSTMGraphData.as_view(), name='lstm-graph-data'),
    path('sarimax/data/', SarimaxGraphData.as_view(), name='sarimax-graph-data'),
    path('plot/', views.plot_data, name='plot_data')  # Updated without the extra 'api/'
]

