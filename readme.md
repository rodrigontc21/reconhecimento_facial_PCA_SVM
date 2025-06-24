# 🧠 Sistema de Reconhecimento Facial com PCA + SVM

Este projeto implementa um sistema de reconhecimento facial utilizando técnicas de **Aprendizado de Máquina**, com foco em **Análise de Componentes Principais (PCA)** para redução de dimensionalidade e **Máquinas de Vetores de Suporte (SVM)** para classificação.

## 🎯 Objetivo
Desenvolver um pipeline completo de reconhecimento facial sobre a base de dados **ORL (Olivetti Research Laboratory)**, contendo 400 imagens de 40 sujeitos diferentes.

---

## 📂 Estrutura do Projeto

```
📁 faces/               ← Pasta com as imagens (não incluída aqui)
├── PCA_SVM.py          ← Script principal com toda a lógica do sistema
├── Relatório.pdf       ← Relatório explicando metodologia e resultados
├── Objetivo.pdf        ← Enunciado da atividade AED3 - IA (PUC Goiás)
└── README.md           ← Este arquivo
```

---

## ⚙️ Tecnologias Utilizadas

- Python 3.x
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imageio`
- `skimage`
- `numpy`
- `TSNE` para projeção dimensional
- `SVM (RBF kernel)` para classificação

---

## 🧪 Funcionalidades

- Leitura e pré-processamento de 400 imagens `.pgm`
- Redução de dimensionalidade com PCA (50 componentes)
- Treinamento de classificador SVM
- Classificação de nova imagem de teste
- Exibição de:
  - Imagem testada e classe prevista
  - Top 5 classes mais prováveis
  - 9 imagens da classe prevista
  - Matriz de confusão (validação cruzada)
  - Projeção t-SNE
  - Análise de erros e limitações

---

## 🧬 Base de Dados

- **ORL (Olivetti Research Laboratory)**
- 40 sujeitos
- 10 imagens por sujeito (total: 400 imagens)
- Imagens com resolução 112x92 pixels, em escala de cinza

---

## 📈 Resultados (Resumo)

- Acurácia de teste acima de 90%
- Classe prevista corretamente para imagem 100.pgm (Sujeito_11)
- Projeção t-SNE mostra boa separabilidade entre classes
- Algumas classes com taxa de erro maior devido à similaridade visual

---

## 📌 Limitações

- Poucas imagens por classe (apenas 10)
- Qualidade da imagem e variações de pose/iluminação afetam a acurácia
- PCA pode perder informação discriminativa
- SVM pode confundir classes visivelmente semelhantes

---

## 📃 Como Executar

1. **Clone este repositório**
```bash
git clone https://github.com/seuusuario/nomedorepositorio.git
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Ajuste o caminho das imagens no `PCA_SVM.py`**
```python
DATA_PATH = r"CAMINHO/DA/PASTA/faces"
```

4. **Execute**
```bash
python PCA_SVM.py
```

---

## 👤 Autor

**Rodrigo F. N. Vieira**  
Disciplina: Inteligência Artificial – AED3  
PUC Goiás – 1º semestre de 2025  
Professor: Clarimar J. Coelho

---

## 📝 Licença

Projeto acadêmico com fins didáticos.  
Uso livre para fins de estudo e extensão educacional.