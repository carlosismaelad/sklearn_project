import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

table = pd.read_csv("./barcos_ref.csv")
print(table)

# Analisar a correlação do preço com os outros indicadores
print(table.corr()[["Preco"]])

# Visualizar a correlação em gráfico (opcional)
sns.heatmap(table.corr()[["Preco"]], cmap="Blues", annot=True)
plt.show()

# Modelagem + algoritmos
# Preparação (Separar a base em X e Y)
y = table["Preco"]
x= table.drop("Preco", axis=1)

# Separando os dados de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Usando as AI's importadas
linear_regression_model = LinearRegression()
decision_tree_model = RandomForestRegressor()

# Repassando a elas os dados de treino
linear_regression_model.fit(x_train, y_train)
decision_tree_model.fit(x_train, y_train)

# Analisando qual dos modelos é o mais preciso
forecast_linear_regression_model = linear_regression_model.predict(x_test)
forecast_decision_tree_model = decision_tree_model.predict(x_test)

print(r2_score(y_test, forecast_linear_regression_model)) # Resultado = 0.4490324760735813
print(r2_score(y_test, forecast_decision_tree_model)) # Resultado = 0.851451444263141

# Interpretação dos resultados
aux_table = pd.DataFrame()
aux_table["y_test"] = y_test
aux_table["Forecast Decision Tree"] = forecast_decision_tree_model
aux_table["Forecast Linear Regression"] = forecast_linear_regression_model
sns.lineplot(aux_table)
plt.show() # Se quiser ver o quanto os dois modelos se aproximam dos dados de preço de teste 

# Importar a tabela com os novos modelos de barcos e fazer a previsão dos preços
new_table = pd.read_csv("./novos_barcos.csv")
forecast = decision_tree_model.predict(new_table)

# Aqui ele exibe a previsão de preços para os novos modelos [Barco1  Barco2 Barco3]
print(forecast)