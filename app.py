from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the new model
model = joblib.load("laptop_price_model.pkl")

@app.route('/api/laptop', methods=['POST'])
def laptop():
    try:
        # รับค่าจากการร้องขอและตรวจสอบค่าที่ได้รับ
        processor_speed = float(request.form.get('processor_speed', 0) or 0)
        ram_size = int(request.form.get('ram_size', 0) or 0)
        storage_capacity = int(request.form.get('storage_capacity', 0) or 0)
        screen_size = float(request.form.get('screen_size', 0) or 0)
        weight = float(request.form.get('weight', 0) or 0)
        
        # เตรียมข้อมูลสำหรับการทำนาย
        x = np.array([[processor_speed, ram_size, storage_capacity, screen_size, weight]])

        # ทำนายโดยใช้โมเดล
        prediction = model.predict(x)

        # ส่งคืนผลการทำนาย
        return {'price': round(prediction[0], 2)}, 200  

    except ValueError:
        return {'error': 'Invalid input values'}, 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
