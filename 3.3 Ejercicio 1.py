import numpy as np
import matplotlib.pyplot as plt

#10x1 −x2 +2x3 = 6
# −1x1 +11x2 −x3 +3x4 = 25
# 2x1 −1x2 +10x3 −1x4 = −11
# −1x2 +3x3 +8x4 −2x5 = 15
# 2x3 −2x4 +10x5 = −10

# Definir el sistema de ecuaciones del ejemplo
A = np.array([
    [10, -1, 2, 0, 0],
    [-1, 11, -1, 3, 0],
    [2, -1, 10, -1, 0],
    [0, -1, 3, 8, -2],
    [0, 0, 2, -2, 10]
])

b = np.array([6, 25, -11, 15, -10])

# Solución exacta para comparar errores
# Se añade esta linea para determinar
# una solución exacta para comparar errores
sol_exacta = np.linalg.solve(A, b) 


# Criterio de paro
tolerancia = 1e-6
max_iter = 100

# Implementación del método de Jacobi
def jacobi(A, b, tol, max_iter):
    n = len(A)
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Calcular errores
        error_abs = np.linalg.norm(x_new - sol_exacta, ord=1)
        error_rel = np.linalg.norm(x_new - sol_exacta, ord=1) / np.linalg.norm(sol_exacta, ord=1)
        error_cuad = np.linalg.norm(x_new - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        # Imprimir errores de la iteración
        print(f"Iteración {k+1}: Error absoluto = {error_abs:.6f}, Error relativo = {error_rel:.6f}, Error cuadrático = {error_cuad:.6f}")
        
        # Criterio de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new
    
    return x, errores_abs, errores_rel, errores_cuad, k+1

# Ejecutar el método de Jacobi
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)

# Graficar los errores
plt.figure(figsize=(8,6))
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Convergencia de los errores en el método de Jacobi")
plt.legend()
plt.grid()
plt.savefig("errores_jacobi.png")  # Guardar la figura en archivo PNG
plt.show()

# Mostrar la solución aproximada
print(f"Solución aproximada: {sol_aprox}")
print(f"Solución exacta: {sol_exacta}")