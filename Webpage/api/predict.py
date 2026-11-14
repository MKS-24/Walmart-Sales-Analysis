import os
import json
import joblib
import pandas as pd
from http.server import BaseHTTPRequestHandler

# Cached model bundle in memory
BUNDLE = None

def get_bundle():
    global BUNDLE
    if BUNDLE is None:
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'best_walmart_model.joblib'),
            os.path.join(os.path.dirname(__file__), '..', 'best_walmart_model.joblib'),
            os.path.abspath('best_walmart_model.joblib'),
            os.path.abspath('api/best_walmart_model.joblib')
        ]
        for path in search_paths:
            if os.path.exists(path):
                BUNDLE = joblib.load(path)
                break
    return BUNDLE

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        b = get_bundle()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()

        response = {
            'status': 'online',
            'model_loaded': b is not None,
            'metrics': b['metrics'] if b else None,
            'message': 'Walmart Sales Forecasting API is running on Vercel Serverless!'
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Empty request body'}).encode('utf-8'))
                return

            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))

            b = get_bundle()
            if not b:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Trained model file could not be loaded on server'}).encode('utf-8'))
                return

            model = b['model']
            feature_cols = b['feature_cols']
            type_mapping = b.get('type_mapping', {'A': 1, 'B': 2, 'C': 3})

            store = int(payload.get('Store', 1))
            dept = int(payload.get('Dept', 1))
            is_holiday = int(payload.get('IsHoliday', 0))
            store_type = str(payload.get('Type', 'A')).strip().upper()
            size = float(payload.get('Size', 151315))

            date_str = str(payload.get('Date', '2012-11-23'))
            try:
                dt = pd.to_datetime(date_str)
                year = int(dt.year)
                month = int(dt.month)
                week = int(dt.isocalendar().week)
                day = int(dt.day)
            except Exception:
                year, month, week, day = 2012, 11, 47, 23

            temp = float(payload.get('Temperature', 60.0))
            fuel = float(payload.get('Fuel_Price', 3.50))
            cpi = float(payload.get('CPI', 215.0))
            unemployment = float(payload.get('Unemployment', 7.5))

            m1 = float(payload.get('MarkDown1', 0.0))
            m2 = float(payload.get('MarkDown2', 0.0))
            m3 = float(payload.get('MarkDown3', 0.0))
            m4 = float(payload.get('MarkDown4', 0.0))
            m5 = float(payload.get('MarkDown5', 0.0))

            store_dept_means = b['store_dept_means']
            dept_means = b['dept_means']
            overall_mean = b['overall_mean']
            historical_avg = store_dept_means.get((store, dept), dept_means.get(dept, overall_mean))

            row_data = {
                'Store': store,
                'Dept': dept,
                'Size': size,
                'Type_Code': type_mapping.get(store_type, 1),
                'IsHoliday': is_holiday,
                'Year': year,
                'Month': month,
                'Week': week,
                'Day': day,
                'Store_Dept_Avg': historical_avg,
                'Temperature': temp,
                'Fuel_Price': fuel,
                'CPI': cpi,
                'Unemployment': unemployment,
                'MarkDown1': m1,
                'MarkDown2': m2,
                'MarkDown3': m3,
                'MarkDown4': m4,
                'MarkDown5': m5,
                'MarkDown1_Active': int(m1 > 0),
                'MarkDown2_Active': int(m2 > 0),
                'MarkDown3_Active': int(m3 > 0),
                'MarkDown4_Active': int(m4 > 0),
                'MarkDown5_Active': int(m5 > 0)
            }

            df_input = pd.DataFrame([row_data])[feature_cols]
            predicted_sales = float(model.predict(df_input)[0])

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()

            result = {
                'status': 'success',
                'predicted_weekly_sales': round(predicted_sales, 2),
                'historical_avg': round(historical_avg, 2),
                'inputs': {
                    'Store': store,
                    'Dept': dept,
                    'Date': date_str,
                    'Type': store_type,
                    'Size': size,
                    'IsHoliday': bool(is_holiday),
                    'Week': week,
                    'Year': year
                }
            }
            self.wfile.write(json.dumps(result).encode('utf-8'))

        except Exception as err:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'message': str(err)}).encode('utf-8'))
