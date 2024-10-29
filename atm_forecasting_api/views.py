# backend/atm_forecasting_api/views.py
from django.shortcuts import render, redirect
import pandas as pd
from django.conf import settings
from atm_forecasting_api.models.XGBoost import xgboost_analysis
from atm_forecasting_api.models.Prophet import prophet_analysis
from atm_forecasting_api.models.LSTM import lstm_analysis
from atm_forecasting_api.models.Sarimax import sarimax_analysis
import os
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import json
from django.http import JsonResponse
import pandas as pd
import os
from django.conf import settings  # Make sure to import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np


@api_view(['GET'])
def get_forecast_data(request):
    dataset = request.GET.get('dataset')

    # Mock example: Load the dataset (replace with your actual dataset handling)
    if dataset == 'atm_cash_demand_single_atm':
        # Example data processing logic (replace with your actual ML model processing)
        rmse_value = 12.34  # Replace with actual RMSE calculation
        forecast_amount = 123456  # Replace with actual forecast value

        return Response({
            'rmse': rmse_value,
            'forecast_amount': forecast_amount
        })

    return Response({'error': 'Dataset not found'}, status=404)

def fetch_data(request):
    dataset1 = request.GET.get('dataset', None)

    # Define the path to your CSV files
    base_path = os.path.join(settings.BASE_DIR, 'Datafile')  # Correct path for Datafile folder

    # Fetch your data based on the dataset_name
    if dataset1:
        try:
            # Load the relevant CSV file
            file_path = os.path.join(base_path, f'{dataset1}.csv')  # This assumes your CSV files are named after the dataset name
            data_frame = pd.read_csv('/home/praba/Desktop/ATM-Cash-Forecasting/Datafile/atm_cash_demand_single_atm.csv')

            # Convert the DataFrame to a dictionary
            data = data_frame.to_dict(orient='records')  # This will give a list of dictionaries
            return JsonResponse(data, safe=False)  # safe=False allows a list response
        except FileNotFoundError:
            return JsonResponse({'error': 'Dataset not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Dataset not specified'}, status=400)


def redirect_model(request):
    if request.method == 'POST':
        selected_model = request.POST.get('model')

        # Logic to redirect to the appropriate analysis view
        if selected_model == 'XGBoost':
            return redirect('xgboost_analysis')  # Redirect to XGBoost analysis view
        elif selected_model == 'Prophet':
            return redirect('prophet_analysis')  # Redirect to Prophet analysis view
        elif selected_model == 'LSTM':
            return redirect('lstm_analysis')  # Redirect to LSTM analysis view
        elif selected_model == 'SARIMAX':
            return redirect('sarimax_analysis')  # Redirect to SARIMAX analysis view

    # If the method is not POST or the selection is invalid, render the index page again
    return render(request, 'index.html', {'error': 'Invalid selection'})


def index(request):
    return render(request, 'index.html')


def xgboost_view(request):
    output_dir = os.path.join('static', 'images')  # Updated output directory

    # Call the xgboost_analysis function
    image_path, rmse = xgboost_analysis(period=30, output_dir=output_dir)

    # Handle case where analysis might fail
    if image_path is None:
        return render(request, 'error.html', {'error_message': 'Error occurred during XGBoost analysis.'})

    # Just send the relative path to the template
    relative_image_path = os.path.relpath(image_path, start='static/')

    return render(request, 'XGBoost_analysis.html', {
        'image_uri': relative_image_path,
        'rmse': rmse
    })


def prophet_view(request):
    output_directory = os.path.join(settings.BASE_DIR, 'static', 'images')  # Define your output directory

    # Call the prophet_analysis function
    image_path = prophet_analysis(
        start_date='2024-01-01',  # Start date for the analysis
        end_date='2025-01-01',    # End date for the analysis
        output_dir=output_directory
    )

    # Render the template and pass the image path
    return render(request, 'prophet_analysis.html', {'image_path': image_path})


def lstm_view(request):
    output_dir = 'path_to_your_output_directory'
    image_path, rmse = lstm_analysis(period=30, output_dir=output_dir)

    if image_path is None:
        return render(request, 'error.html', {'error_message': 'Error occurred during LSTM analysis.'})

    return render(request, 'LSTM_analysis.html', {
        'image_path': image_path,
        'rmse': rmse
    })


from django.shortcuts import render
import os

def sarimax_view(request):
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    output_dir = os.path.join(settings.BASE_DIR, 'static', 'images')

    # Call your SARIMAX analysis function
    image_path = sarimax_analysis(start_date, end_date, output_dir)

    # Render the template to display the graph
    return render(request, 'sarimax_analysis.html', {'image_path': image_path})




class XGBoostGraphData(APIView):
    def get(self, request):
        # Replace this with your actual logic to generate or retrieve the graph data
        labels = ["Jan", "Feb", "Mar", "Apr"]  # Replace with your actual labels
        values = [20, 30, 40, 50]  # Replace with your actual data values

        data = {
            "labels": labels,
            "values": values
        }
        return Response(data)


class ProphetGraphData(APIView):
    def get(self, request):
        # Replace this with your actual logic to generate or retrieve the graph data for Prophet
        labels = ["Jan", "Feb", "Mar", "Apr"]
        values = [10, 20, 30, 40]

        data = {
            "labels": labels,
            "values": values
        }
        return Response(data)


class LSTMGraphData(APIView):
    def get(self, request):
        # Replace this with your actual logic to generate or retrieve the graph data for LSTM
        labels = ["Jan", "Feb", "Mar", "Apr"]
        values = [15, 25, 35, 45]

        data = {
            "labels": labels,
            "values": values
        }
        return Response(data)


class SarimaxGraphData(APIView):
    def get(self, request):
        # Replace this with your actual logic to generate or retrieve the graph data for SARIMAX
        labels = ["Jan", "Feb", "Mar", "Apr"]
        values = [12, 22, 32, 42]

        data = {
            "labels": labels,
            "values": values
        }
        return Response(data)

from django.http import JsonResponse
import json

def plot_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        dataset = data.get('dataset', None)
        if dataset:
            # Process the dataset and return a response
            return JsonResponse({'status': 'success', 'message': f'Dataset {dataset} processed.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No dataset provided.'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

