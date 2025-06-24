import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionSystem:
    def __init__(self, data_path, n_components=50):
        """
        Inicializa o sistema de reconhecimento facial.
        
        Args:
            data_path (str): Caminho para a pasta contendo as imagens
            n_components (int): Número de componentes principais para PCA
        """
        self.data_path = data_path
        self.n_components = n_components
        self.image_size = (112, 92)  # Padrão ORL
        self.pca = None
        self.svm = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_vectors = None
        self.labels = None
        self.class_names = []
        
    def load_orl_dataset(self):
        """
        Carrega a base de dados ORL com estrutura:
        - 400 imagens numeradas: 0.pgm até 399.pgm
        - 40 sujeitos com 10 imagens cada (0-9: sujeito 1, 10-19: sujeito 2, etc.)
        """
        images = []
        labels = []
        
        print("Carregando base de dados ORL...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Pasta não encontrada: {self.data_path}")
        
        # Carrega arquivos de 0.pgm até 399.pgm
        for i in range(400):
            filename = f"{i}.pgm"
            img_path = os.path.join(self.data_path, filename)
            
            if os.path.exists(img_path):
                try:
                    # Determina o sujeito: imagens 0-9 = sujeito 0, 10-19 = sujeito 1, etc.
                    current_subject = i // 10
                    
                    # Carrega e processa a imagem
                    img = imageio.imread(img_path)
                    img = resize(img, self.image_size, anti_aliasing=True)
                    img_array = np.array(img).flatten()
                    
                    images.append(img_array)
                    labels.append(current_subject)
                    
                    if i % 50 == 0:
                        print(f"Processadas {i+1} imagens...")
                        
                except Exception as e:
                    print(f"Erro ao carregar {filename}: {e}")
            else:
                print(f"Arquivo não encontrado: {filename}")
        
        # Converte para arrays numpy
        self.feature_vectors = np.array(images)
        self.labels = np.array(labels)
        
        # Cria nomes das classes
        self.class_names = [f"Sujeito_{i+1:02d}" for i in range(len(np.unique(self.labels)))]
        
        print(f"Base carregada: {len(images)} imagens de {len(np.unique(labels))} sujeitos")
        print(f"Distribuição: {len(images)//len(np.unique(labels))} imagens por sujeito")
        return self.feature_vectors, self.labels
    
    def preprocess_data(self):
        """
        Preprocessa os dados aplicando normalização.
        """
        print("Preprocessando dados...")
        
        # Normaliza os dados
        self.feature_vectors = self.scaler.fit_transform(self.feature_vectors)
        
        # Divide em treino e teste (80% treino, 20% teste)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feature_vectors, self.labels, test_size=0.3, 
            stratify=self.labels, random_state=42
        )
        
        print(f"Dados de treino: {self.X_train.shape}")
        print(f"Dados de teste: {self.X_test.shape}")
    
    def apply_pca(self):
        """
        Aplica PCA para redução de dimensionalidade.
        """
        print(f"Aplicando PCA com {self.n_components} componentes...")
        
        self.pca = PCA(n_components=self.n_components, random_state=42)
        
        # Treina PCA apenas com dados de treino
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)
        
        # Calcula variância explicada
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"Variância explicada pelo PCA: {explained_variance:.4f}")
        
        return self.X_train_pca, self.X_test_pca
    
    def train_svm(self):
        """
        Treina o classificador SVM com kernel RBF.
        """
        print("Treinando classificador SVM...")
        
        # Treina SVM com kernel RBF
        self.svm = SVC(kernel='rbf', probability=True, random_state=42, gamma='scale')
        self.svm.fit(self.X_train_pca, self.y_train)
        
        # Avalia no conjunto de teste
        accuracy = self.svm.score(self.X_test_pca, self.y_test)
        print(f"Acurácia no teste: {accuracy:.4f}")
        
        return self.svm
    
    def classify_image(self, test_image_path):
        """
        Classifica uma imagem de teste.
        
        Args:
            test_image_path (str): Caminho para a imagem de teste
            
        Returns:
            tuple: (classe_prevista, probabilidades, imagem_processada)
        """
        # Carrega e processa a imagem de teste
        test_img = imageio.imread(test_image_path)
        test_img = resize(test_img, self.image_size, anti_aliasing=True)
        test_img_array = np.array(test_img).flatten().reshape(1, -1)
        
        # Normaliza
        test_img_normalized = self.scaler.transform(test_img_array)
        
        # Aplica PCA
        test_img_pca = self.pca.transform(test_img_normalized)
        
        # Classifica
        predicted_class = self.svm.predict(test_img_pca)[0]
        probabilities = self.svm.predict_proba(test_img_pca)[0]
        
        return predicted_class, probabilities, test_img
    
    def get_class_images(self, class_id, n_images=9):
        """
        Retorna n imagens da classe especificada.
        """
        # Encontra índices da classe
        class_indices = np.where(self.labels == class_id)[0]
        
        # Seleciona n imagens aleatórias da classe
        selected_indices = np.random.choice(class_indices, 
                                          min(n_images, len(class_indices)), 
                                          replace=False)
        
        images = []
        for idx in selected_indices:
            # Reconstrói a imagem a partir do vetor de características
            img_vector = self.feature_vectors[idx]
            # Denormaliza
            img_vector = self.scaler.inverse_transform(img_vector.reshape(1, -1))[0]
            img = img_vector.reshape(self.image_size)
            images.append(img)
        
        return images
    
    def plot_classification_results(self, test_image_path):
        """
        Plota os resultados da classificação.
        """
        # Classifica a imagem de teste
        predicted_class, probabilities, test_img = self.classify_image(test_image_path)
        
        # Obtém top-5 classes
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_probs = probabilities[top5_indices]
        top5_classes = [self.class_names[i] for i in top5_indices]
        
        # Obtém imagens da classe prevista
        class_images = self.get_class_images(predicted_class)
        
        # Cria a figura
        fig = plt.figure(figsize=(20, 12))

        # Subplot 1: imagem de teste
        plt.subplot(3, 4, 1)
        plt.imshow(test_img, cmap='gray')
        plt.title(f'Imagem de Teste\nClasse Prevista: {self.class_names[predicted_class]}', 
                fontsize=12, fontweight='bold')
        plt.axis('off')

        # Subplot 2: top-5 classes
        plt.subplot(3, 4, 2)
        bars = plt.barh(range(5), top5_probs)
        plt.yticks(range(5), top5_classes)
        plt.xlabel('Probabilidade')
        plt.title('Top-5 Classes Mais Prováveis', fontweight='bold')
        plt.gca().invert_yaxis()
        for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center')

        # Subplots 3 a 11: 9 imagens da classe prevista
        for i, img in enumerate(class_images[:9]):
            plt.subplot(3, 4, i + 3)  # agora suporta até subplot(3, 4, 11)
            plt.imshow(img, cmap='gray')
            plt.title(f'{self.class_names[predicted_class]} - Img {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return predicted_class, top5_classes, top5_probs
    
    def cross_validation_analysis(self):
        """
        Realiza validação cruzada e mostra matriz de confusão.
        """
        print("Realizando validação cruzada 5-fold...")
        
        # Validação cruzada
        cv_scores = cross_val_score(self.svm, self.X_train_pca, self.y_train, 
                                  cv=5, scoring='accuracy')
        
        print(f"Scores CV: {cv_scores}")
        print(f"Acurácia média CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Matriz de confusão no conjunto de teste
        y_pred = self.svm.predict(self.X_test_pca)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plota matriz de confusão
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusão - Conjunto de Teste', fontweight='bold')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Prevista')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.class_names))
        
        return cv_scores, cm
    
    def plot_tsne(self):
        """
        Plota projeção t-SNE dos vetores PCA coloridos por classe.
        """
        print("Gerando projeção t-SNE...")
        
        # Combina dados de treino e teste para visualização completa
        X_all_pca = np.vstack([self.X_train_pca, self.X_test_pca])
        y_all = np.hstack([self.y_train, self.y_test])
        
        # Aplica t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                   n_iter=1000, learning_rate=200)
        X_tsne = tsne.fit_transform(X_all_pca)
        
        # Plota
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_all, 
                            cmap='tab20', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Classe')
        plt.title('Projeção t-SNE dos Vetores PCA', fontweight='bold')
        plt.xlabel('Componente t-SNE 1')
        plt.ylabel('Componente t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self):
        """
        Analisa os erros do modelo e características das classes.
        """
        print("\n=== ANÁLISE DE RESULTADOS ===")
        
        # Predições no conjunto de teste
        y_pred = self.svm.predict(self.X_test_pca)
        
        # Encontra erros
        errors = np.where(self.y_test != y_pred)[0]
        
        print(f"\nTotal de erros no teste: {len(errors)}/{len(self.y_test)}")
        print(f"Taxa de erro: {len(errors)/len(self.y_test):.4f}")
        
        # Analisa erros por classe
        error_analysis = {}
        for i in range(len(self.class_names)):
            class_indices = np.where(self.y_test == i)[0]
            class_errors = np.intersect1d(class_indices, errors)
            error_rate = len(class_errors) / len(class_indices) if len(class_indices) > 0 else 0
            error_analysis[i] = {
                'total': len(class_indices),
                'errors': len(class_errors),
                'error_rate': error_rate
            }
        
        # Classes com mais erros
        worst_classes = sorted(error_analysis.items(), 
                             key=lambda x: x[1]['error_rate'], reverse=True)[:5]
        
        print("\nClasses com maiores taxas de erro:")
        for class_id, stats in worst_classes:
            print(f"{self.class_names[class_id]}: "
                  f"{stats['error_rate']:.4f} ({stats['errors']}/{stats['total']})")
        
        # Separabilidade entre classes
        print(f"\nSeparabilidade entre classes:")
        print(f"Número de componentes PCA: {self.n_components}")
        print(f"Variância explicada: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Limitações do modelo
        print(f"\nLimitações identificadas:")
        print(f"1. Redução de dimensionalidade pode perder informações discriminativas")
        print(f"2. SVM pode ter dificuldade com classes muito similares")
        print(f"3. Qualidade das imagens afeta o desempenho")
        print(f"4. Número limitado de amostras por classe ({len(self.feature_vectors)//len(self.class_names)} por classe)")


# Exemplo de uso
if __name__ == "__main__":
    # Configuração - AJUSTE ESTE CAMINHO PARA O SEU SISTEMA
    # Exemplo: r"C:\Users\kazee\OneDrive\Desktop\Python\IA Claurimar\faces"
    DATA_PATH = 
    
    # Verificação rápida do caminho
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Pasta não encontrada: {DATA_PATH}")
        exit()
    
    # Verifica se tem os arquivos esperados
    test_files = [f"{i}.pgm" for i in range(5)]  # Testa primeiros 5
    missing_files = [f for f in test_files if not os.path.exists(os.path.join(DATA_PATH, f))]
    
    if missing_files:
        print(f"AVISO: Alguns arquivos não foram encontrados: {missing_files}")
        print("Continuando mesmo assim...")
    else:
        print("Caminho e arquivos verificados com sucesso!")
    
    # Cria o sistema
    face_system = FaceRecognitionSystem(DATA_PATH, n_components=50)
    
    # Pipeline completo
    print("\n=== SISTEMA DE RECONHECIMENTO FACIAL ===")
    
    try:
        # 1. Carrega dados
        face_system.load_orl_dataset()
        
        # 2. Preprocessa
        face_system.preprocess_data()
        
        # 3. Aplica PCA
        face_system.apply_pca()
        
        # 4. Treina SVM
        face_system.train_svm()
        
        # 5. Validação cruzada e matriz de confusão
        face_system.cross_validation_analysis()
        
        # 6. Projeção t-SNE
        face_system.plot_tsne()
        
        # 7. Análise de erros
        face_system.analyze_errors()
        
        # 8. Teste com imagens específicas
        print("\n=== TESTANDO CLASSIFICAÇÃO ===")
        
        # Testa com algumas imagens diferentes
        test_images = [15, 67, 123, 289, 345]  # Diferentes sujeitos
        
        # Exemplo: r"C:\Users\kazee\OneDrive\Desktop\Python\IA Claurimar\faces\10.pgm""
        test_image_path = 
        
        
        predicted_class, top5_classes, top5_probs = face_system.plot_classification_results(test_image_path)
        
        print(f"Imagem testada: {test_image_path}")
        print(f"Classe prevista: {predicted_class}")
        print(f"Classes mais prováveis: {top5_classes}")
        
        print(f"Confiança: {top5_probs[0]:.3f}")
        print("-" * 50)
    
        print("\n=== PIPELINE CONCLUÍDO ===")
        
    except Exception as e:
        print(f"ERRO durante execução: {e}")
        print("Verifique se:")
        print("1. O caminho está correto")
        print("2. Os arquivos .pgm existem")
        print("3. As bibliotecas estão instaladas")