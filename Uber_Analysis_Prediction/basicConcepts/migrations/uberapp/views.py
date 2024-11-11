from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from tensorflow.keras.models import load_model

# Load the model once at the module level to avoid reloading on each request
model = load_model('savedmodels/model.h5')

def predictor(request):
    return render(request, 'main.html')

def forminfo(request):
    if request.method == 'GET':
        try:
            # Extract all required features from the request
            features = [
                int(request.GET['Car_Condition']),
                int(request.GET['Weather']),
                int(request.GET['Traffic_Condition']),
                float(request.GET['pickup_longitude']),
                float(request.GET['pickup_latitude']),
                float(request.GET['dropoff_longitude']),
                float(request.GET['dropoff_latitude']),
                int(request.GET['passenger_count']),
                int(request.GET['hour']),
                int(request.GET['day']),
                int(request.GET['month']),
                int(request.GET['weekday']),
                int(request.GET['year']),
                float(request.GET['jfk_dist']),
                float(request.GET['ewr_dist']),
                float(request.GET['lga_dist']),
                float(request.GET['sol_dist']),
                float(request.GET['nyc_dist']),
                float(request.GET['distance']),
                float(request.GET['bearing'])
            ]

            # Convert features to a DataFrame for prediction
            feature_names = ['Car_Condition', 'Weather', 'Traffic_Condition',
                             'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                             'dropoff_latitude', 'passenger_count', 'hour', 'day',
                             'month', 'weekday', 'year', 'jfk_dist', 'ewr_dist',
                             'lga_dist', 'sol_dist', 'nyc_dist', 'distance', 'bearing']
                             
            input_data = pd.DataFrame([features], columns=feature_names)

            # Make the prediction
            prediction = model.predict(input_data)

            # Convert prediction to a native Python float
            predicted_fare_amount = float(prediction[0][0])

            # Render the result page with the predicted fare amount
            return render(request, 'result.html', {'result': predicted_fare_amount})

        except Exception as e:
            return render(request, 'result.html', {'result': f'Error: {str(e)}'})
    
    return render(request, 'result.html', {'result': 'Invalid request method.'})
