# Neural-Net  
A simple neural network implemented from scratch in Java.

## 📌 Overview
This project is a lightweight, from-scratch implementation of a feed-forward artificial neural network (ANN) written entirely in **Java**, with **no external machine-learning libraries**.

It is designed for learning, experimentation, and understanding the internal mechanics of neural networks, including:

- Neurons and weighted connections  
- Forward propagation  
- Backpropagation  
- Gradient descent training  
- Activation functions  

The goal of the project is to demonstrate how neural networks work at a low level, providing a transparent and modifiable implementation.

---

## 🚀 Features
- Fully custom implementation of:
  - Layers and neurons  
  - Feed-forward computation  
  - Backpropagation for training  
  - Weight and bias updates using gradient descent  
- Support for common activation functions (e.g., sigmoid, tanh, ReLU — depending on implementation)  
- Flexible network architecture (choose layer sizes)  
- No dependencies beyond standard Java  
- Clear, beginner-friendly structure  

---

## 🧠 How It Works

### 1. **Network Structure**
You can define:
- Number of input neurons  
- Number and size of hidden layers  
- Number of outputs  

### 2. **Forward Propagation**
The network computes predictions by:
- Multiplying inputs by weights  
- Adding biases  
- Applying an activation function  

### 3. **Backpropagation**
During training:
- The error is computed (usually Mean Squared Error)  
- Gradients are calculated layer by layer  
- Weights and biases are adjusted to minimize the loss  

### **🎯 Goals**
- Demonstrate ANN fundamentals clearly
- Serve as a learning tool for students and developers
- Provide a base for experimenting with neural-network extensions (improved activations, optimizers, etc.)

### 📜 License
MIT License — feel free to use, modify, and learn from this project.

