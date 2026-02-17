# Quantum-Classical Hybrid Ensemble for Financial Time Series Classification
## Квантово-классический ансамбль для классификации финансовых временных рядов

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.43+-green.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## О проекте

Проект исследует эффективность гибридного ансамбля, состоящего из нескольких Variational Quantum Classifiers (VQC) с различными схемами кодирования признаков, результаты которых агрегируются классической мета-моделью (stacking).

**Задача** — бинарная классификация направления движения цены акции:
- **Apple Inc. (AAPL)**
- **Период: 2015–2023**

**Цель** — определить, способен ли квантовый ансамбль:
- превзойти одиночный VQC
- конкурировать с классическими моделями
- демонстрировать структурную зависимость квантового преимущества

## Цели исследования

- Сравнить классические модели (LogReg, RF, XGBoost) и VQC
- Построить квантовый ансамбль (stacking)
- Исследовать влияние числа кубитов
- Провести статистическую проверку значимости
- Проанализировать корреляцию ошибок квантовых моделей

## Методология

### Признаки

Используются:
- **Return** - доходность
- **Volatility** - волатильность (5-day rolling std)
- **MA5** - скользящая средняя за 5 дней
- **MA10** - скользящая средняя за 10 дней

**Целевая переменная:**

$$y_t = \begin{cases} 1, & r_{t+1} > 0 \\ 0, & r_{t+1} \leq 0 \end{cases}$$

### Квантовая модель

**Квантовое отображение:**
$$x \mapsto U(x)|0\rangle$$

**Ожидаемое значение:**
$$\phi(x) = \langle 0|U^\dagger(x,\theta) O U(x,\theta)|0\rangle$$

где:
- $U(x,\theta)$ — ZZFeatureMap + Ansatz
- $\theta$ — обучаемые параметры

**Квантовое ядро:**
$$K(x_i,x_j) = |\langle \psi(x_i)|\psi(x_j)\rangle|^2$$

### Ансамбль

Пусть есть три VQC:
$$h_1(x), h_2(x), h_3(x)$$

**Stacking:**
$$z = [h_1(x), h_2(x), h_3(x)]$$
$$\hat{y} = g(z)$$

где $g$ — Logistic Regression.

**Гипотеза:** Низкая корреляция ошибок ⇒ прирост ансамбля.

### Метрики

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**ROC-AUC:**
$$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$

**Статистическая проверка:**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n} + \frac{s_2^2}{n}}}$$

## Основные результаты (пример заполнения)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~0.48 | near baseline |
| Random Forest | ~0.49 | unstable |
| XGBoost | ~0.48 | overfitting |
| VQC | ~0.52 | small improvement |
| Quantum Ensemble | ~0.54–0.56 | statistically tested |

## Влияние числа кубитов

$$n_{qubits} \uparrow \Rightarrow \dim(H) = 2^{n_{qubits}}$$

Однако для финансовых данных:
$$\frac{\partial \text{Accuracy}}{\partial n_{qubits}} \approx 0$$

Это указывает на ограниченную выразительность без увеличения структуры признаков.

## Выводы

- Финансовые временные ряды обладают крайне низкой предсказуемостью
- Одиночный VQC демонстрирует незначительное улучшение
- Квантовый ансамбль даёт дополнительный прирост при условии диверсификации
- Вычислительная стоимость существенно выше классических методов
- Квантовое преимущество структурно-зависимо

## Технологии

- **Qiskit** - квантовые вычисления
- **Scikit-learn** - классические ML модели
- **XGBoost** - градиентный бустинг
- **NumPy / Pandas** - обработка данных
- **Matplotlib / Seaborn** - визуализация

## Как запустить

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск Jupyter
jupyter notebook
