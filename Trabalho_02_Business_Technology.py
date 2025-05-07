
# Projeto: Previsão de Vendas com Machine Learning para Empresa XYZ

# Estrutura do Projeto:
# 1. Modelo de Machine Learning para Previsão de Vendas (diário, semanal, mensal)
# 2. API em Flask para comunicação com Frontend
# 3. Frontend em HTML, CSS e JS para visualização dos dados
# 4. Elementos de Gamificação
# 5. Integração com Pagamento (simulado)
# 6. Deploy no Vercel

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from flask import Flask, jsonify, request

data = pd.DataFrame({
    'data': pd.date_range(start='2024-01-01', periods=100),
    'vendas': np.random.randint(100, 200, size=100)
})

data['dia'] = data['data'].dt.day
data['mes'] = data['data'].dt.month
data['ano'] = data['data'].dt.year

X = data[['dia', 'mes', 'ano']]
y = data['vendas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    prediction = model.predict(df[['dia', 'mes', 'ano']])
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
