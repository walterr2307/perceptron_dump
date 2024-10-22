# Importa as funções randint e uniform do módulo random
from random import randint, uniform
from pandas import read_csv  # Importa a função read_csv do módulo pandas
# Importa o classificador KNN do scikit-learn
from sklearn.neighbors import KNeighborsClassifier
# Importa o classificador Decision Tree do scikit-learn
from sklearn.tree import DecisionTreeClassifier
# Importa o classificador Random Forest do scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import \
    train_test_split  # Importa a função train_test_split do scikit-learn para dividir os dados em treino e teste
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC  # Importa o classificador SVM do scikit-learn
from sklearn.preprocessing import LabelEncoder  # Importa o LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd  # Importa pandas como pd
import matplotlib.pyplot as plt  # Importa matplotlib para visualização
from sklearn.feature_selection import f_classif


# Função para somar dois vetores, com uma condição específica para a segunda posição
def somar(parcela1, parcela2, tipo_algoritmo):
    soma = []
    for i in range(len(parcela1)):
        if tipo_algoritmo > 0 and i == 1:  #
            # Arredonda a soma para 4 casas decimais
            soma.append(round(parcela1[i] + parcela2[i], 4))
        else:
            soma.append(parcela1[i] + parcela2[i])
    return soma


# Função para subtrair dois vetores
def subtrair(minuendo, subtraendo):
    sub = []
    for i in range(len(minuendo)):
        sub.append(minuendo[i] - subtraendo[i])
    return sub


# Função para multiplicar um vetor por uma constante, com uma condição específica para a segunda posição
def multiplicar(vetor, c, tipo_algoritmo):
    mult = []
    for i in range(len(vetor)):
        if tipo_algoritmo > 0 and i == 1:  # Se o tipo de algoritmo for maior que 0 e estiver na segunda posição
            # Multiplica o valor pelo coeficiente e adiciona à lista
            mult.append(vetor[i] * c)
        else:
            # Arredonda o resultado da multiplicação e adiciona à lista
            mult.append(round(vetor[i] * c))
    return mult


# Classe para representar uma partícula no algoritmo PSO
class Particula:
    def __init__(self, pos, vel):
        self.pos = pos  # Posição da partícula
        self.vel = vel  # Velocidade da partícula
        self.perf = None  # Desempenho da partícula
        self.c1 = uniform(0, 2)  # Coeficiente de aprendizado 1
        self.c2 = uniform(0, 2)  # Coeficiente de aprendizado 2
        self.melhor_pos = None  # Melhor posição encontrada pela partícula
        self.melhor_perf = None  # Melhor desempenho encontrado pela partícula

    def __str__(self):
        return 'Position: {} --> Speed: {} --> Porcentage: {:.1f}%'.format(self.pos, self.vel, self.perf)

    # Método para atualizar a velocidade da partícula
    def prox_vel(self, melhor_pos_geral, tipo_algoritmo):
        sub1 = subtrair(self.melhor_pos,
                        self.pos)  # Vetor de diferença entre a posição atual e a melhor posição individual
        sub2 = subtrair(melhor_pos_geral,
                        self.pos)  # Vetor de diferença entre a posição atual e a melhor posição global
        mult1 = multiplicar(sub1, self.c1,
                            tipo_algoritmo)  # Multiplica o vetor de diferença 1 pelo coeficiente de aprendizado 1
        mult2 = multiplicar(sub2, self.c2,
                            tipo_algoritmo)  # Multiplica o vetor de diferença 2 pelo coeficiente de aprendizado 2
        soma = somar(mult1, mult2, tipo_algoritmo)  # Soma os dois resultados
        # Atualiza a velocidade da partícula
        self.vel = somar(soma, self.vel, tipo_algoritmo)


# Classe para implementar o algoritmo PSO
class PSO:
    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo, endereco_csv, metrica):
        # Tamanho do enxame (número de partículas)
        self.tam_enxame = tam_enxame
        self.num_interacoes = num_interacoes  # Número de iterações do algoritmo
        self.tipo_algoritmo = tipo_algoritmo  # Tipo de algoritmo a ser utilizado
        # Dataframe carregado a partir do arquivo CSV
        self.dataframe = read_csv(endereco_csv)
        self.metrica = metrica
        self.label_encoder = LabelEncoder()  # Inicializa o LabelEncoder
        self.melhor_pos_geral = None  # Melhor posição global encontrada pelo algoritmo
        self.melhor_perf_geral = 0  # Melhor desempenho global encontrado pelo algoritmo

        # Mapeamento dos parâmetros de acordo com o tipo de algoritmo
        self.weights_map = {0: 'uniform', 1: 'distance'}
        self.criterion_map = {0: 'entropy', 1: 'gini', 2: 'log_loss'}
        self.kernel_map = {0: 'sigmoid', 1: 'poly', 2: 'rbf', 3: 'linear'}
        self.solver_map = {0: 'newton-cg', 1: 'lbfgs',
                           2: 'liblinear', 3: 'sag', 4: 'saga'}

        # Valores mínimos e máximos dos parâmetros de acordo com o tipo de algoritmo
        self.min = [[1, 0], [3, 0.1, 0], [3, 0.1, 0, 3],
                    [0, 0.1], [100, 0.1, 0], [3, 0.1, 0, 3]]
        self.max = [[20, 1], [20, 0.3, 2], [20, 0.3, 2, 200],
                    [3, 1], [200, 1, 4], [20, 0.3, 2, 200]]
        self.executar()  # Executa o algoritmo

    # Método para definir os conjuntos de dados de treinamento e teste
    def definir_xy(self):
        num_colunas = len(self.dataframe.columns)

        # Identifica as colunas categóricas
        # Lista de colunas categóricas a serem codificadas
        colunas_categoricas = ['Category']

        # Aplica Label Encoding nas colunas categóricas
        for coluna in colunas_categoricas:
            self.dataframe[coluna] = self.label_encoder.fit_transform(
                self.dataframe[coluna])

        # Define X (variáveis independentes) e Y (variável dependente)
        x = self.dataframe.iloc[:, 0:num_colunas - 1].values
        y = self.dataframe.iloc[:, num_colunas - 1].values

        # Aplica LabelEncoder à variável alvo (y)
        y = self.label_encoder.fit_transform(y)

        # Divide os dados em conjuntos de treino e teste
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = train_test_split(x, y, test_size=0.25,
                                                                                              random_state=0)

    # Método para gerar uma posição inicial para as partículas
    def pos_zero(self):
        zero = []
        for i in range(len(self.max[self.tipo_algoritmo])):
            zero.append(self.max[self.tipo_algoritmo][i] // 2)
        return zero

    # Método para gerar uma posição aleatória para as partículas de acordo com o tipo de algoritmo
    def pos_aleatoria(self):
        if self.tipo_algoritmo == 0:
            return [randint(1, 20), randint(0, 1)]
        elif self.tipo_algoritmo == 1:
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2)]
        elif self.tipo_algoritmo == 2:
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2), randint(3, 200)]
        elif self.tipo_algoritmo == 3:
            return [randint(0, 3), round(uniform(0.1, 1), 4)]
        elif self.tipo_algoritmo == 4:
            return [randint(100, 200), round(uniform(0.1, 1), 4), randint(0, 4)]
        else:
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2), randint(3, 200)]

    # Método para gerar uma partícula com posição e velocidade aleatórias
    def gerar_particula(self):
        pos = self.pos_aleatoria()
        vel = subtrair(pos, self.pos_zero())
        p = Particula(pos, vel)
        p.melhor_pos = list(p.pos)
        p.perf = self.fitness(p)
        p.melhor_perf = p.perf
        return p

    # Método para gerar um modelo de acordo com a posição da partícula
    def gerar_modelo(self, p):
        if self.tipo_algoritmo == 0:
            return KNeighborsClassifier(n_neighbors=p.pos[0], weights=self.weights_map[p.pos[1]])
        elif self.tipo_algoritmo == 1:
            return DecisionTreeClassifier(max_depth=p.pos[0], min_samples_split=p.pos[1],
                                          criterion=self.criterion_map[p.pos[2]])
        elif self.tipo_algoritmo == 2:
            return RandomForestClassifier(max_depth=p.pos[0], min_samples_split=p.pos[1],
                                          criterion=self.criterion_map[p.pos[2]], n_estimators=p.pos[3])
        elif self.tipo_algoritmo == 3:
            return SVC(kernel=self.kernel_map[p.pos[0]], C=p.pos[1])
        elif self.tipo_algoritmo == 4:
            return LogisticRegression(max_iter=p.pos[0], C=p.pos[1], solver=self.solver_map[p.pos[2]])
        else:
            return ExtraTreesClassifier(max_depth=p.pos[0], min_samples_split=p.pos[1],
                                        criterion=self.criterion_map[p.pos[2]], n_estimators=p.pos[3])

    def fitness(self, p, usar_anova=False):
        if usar_anova:
            # Aplicar ANOVA
            F_values, p_values = f_classif(
                self.x_treinamento, self.y_treinamento)
            features_significativas = [i for i in range(
                len(p_values)) if p_values[i] < 0.05]
            x_treinamento_filtrado = self.x_treinamento[:,
                                                        features_significativas]
            x_teste_filtrado = self.x_teste[:, features_significativas]
        else:
            x_treinamento_filtrado = self.x_treinamento
            x_teste_filtrado = self.x_teste

        # Gerar e treinar o modelo
        modelo = self.gerar_modelo(p)
        modelo.fit(x_treinamento_filtrado, self.y_treinamento)
        previsoes = modelo.predict(x_teste_filtrado)

        # Para AUC, você precisa das probabilidades das classes
        probabilidades = modelo.predict_proba(x_teste_filtrado)[
            :, 1]  # Probabilidades da classe positiva

        # Calcular as métricas
        if self.metrica == 0:
            perf = accuracy_score(self.y_teste, previsoes)
        elif self.metrica == 1:
            perf = precision_score(self.y_teste, previsoes, average='weighted')
        elif self.metrica == 2:
            perf = recall_score(self.y_teste, previsoes, average='weighted')
        elif self.metrica == 3:
            perf = f1_score(self.y_teste, previsoes, average='weighted')
        else:
            perf = roc_auc_score(self.y_teste, probabilidades)

        return perf * 100

    # Método para gerar o enxame inicial
    def gerar_enxame(self):
        enxame = []
        for i in range(self.tam_enxame):
            p = self.gerar_particula()
            enxame.append(p)
            if p.melhor_perf > self.melhor_perf_geral:
                self.melhor_perf_geral = p.melhor_perf
                self.melhor_pos_geral = list(p.melhor_pos)
        return enxame

    # Método para mover as partículas e atualizar suas posições e velocidades
    def mover(self, p):
        p.prox_vel(self.melhor_pos_geral, self.tipo_algoritmo)
        p.pos = somar(p.pos, p.vel, self.tipo_algoritmo)

        for i in range(len(self.max[self.tipo_algoritmo])):
            if p.pos[i] > self.max[self.tipo_algoritmo][i]:
                p.pos[i] = self.max[self.tipo_algoritmo][i]
            if p.pos[i] < self.min[self.tipo_algoritmo][i]:
                p.pos[i] = self.min[self.tipo_algoritmo][i]

        p.perf = self.fitness(p)

        if p.perf > p.melhor_perf:
            p.melhor_perf = p.perf
            p.melhor_pos = list(p.pos)
        if p.melhor_perf > self.melhor_perf_geral:
            self.melhor_perf_geral = p.melhor_perf
            self.melhor_pos_geral = list(p.melhor_pos)

    def calcular_importancia_features(self):
        # Gerar o modelo com os melhores parâmetros encontrados
        modelo = self.gerar_modelo(Particula(self.melhor_pos_geral, [0] * len(
            self.melhor_pos_geral)))  # Cria uma partícula com a melhor posição
        # Treina o modelo com os dados de treinamento
        modelo.fit(self.x_treinamento, self.y_treinamento)

        # Obter a importância das features
        importances = modelo.feature_importances_

        # Criar um DataFrame para visualizar
        feature_importance_df = pd.DataFrame(
            {'Feature': self.dataframe.columns[:-1], 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)

        # Exibir a importância das features
        print("\nImportância das Features:")
        print(feature_importance_df)

        # Visualização da importância das features
        plt.figure(figsize=(10, 22))
        plt.barh(feature_importance_df['Feature'],
                 feature_importance_df['Importance'])
        plt.xlabel('Importância')
        plt.title('Importância das Features - Random Forest')
        # Inverte o eixo y para mostrar a feature mais importante no topo
        plt.gca().invert_yaxis()
        plt.show()

    # Método para executar o PSO e calcular a importância das features
    def executar(self):
        self.definir_xy()
        enxame = self.gerar_enxame()

        with open("PSO.txt", "w") as file:
            file.write("1* Iteration:\n")

            for i in range(self.tam_enxame):
                file.write(str(enxame[i]) + '\n')

            for j in range(1, self.num_interacoes):
                file.write("\n{}* Iteration:\n".format(j + 1))

                for i in range(self.tam_enxame):
                    self.mover(enxame[i])
                    file.write(str(enxame[i]) + '\n')

            file.write(
                '\nBest Position: {}\nBest Porcentage: {:.1f}%'.format(self.melhor_pos_geral, self.melhor_perf_geral))

        # Chama a função para calcular a importância das features
        # self.calcular_importancia_features()


# Criando uma instância da classe PSO com os parâmetros específicos
pso = PSO(10, 0, 1, '/content/Obfuscated-MalMem2022.csv', 3)
