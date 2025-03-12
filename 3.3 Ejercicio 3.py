import numpy as np
import matplotlib.pyplot as plt

#12x1 −2x2 +x3 = 20
# −3x1 +18x2 −4x3 +2x4 = 35
# x1 −2x2 +16x3 −x4 +x5 = −5
# 2x2 −x3 +11x4 −3x5 +x6 = 19
# −2x3 +4x4 +15x5 −2x6 +x7 = −12
# x4 −3x5 +2x6 +13x7 = 25

# Definir el sistema de ecuaciones del ejemplo
A = np.array([
    [12, -2, 1, 0, 0, 0, 0],
    [-3, 18, -4, 2, 0, 0, 0],
    [1, -2, 16, -1, 1, 0, 0],
    [0, 2, -1, 11, -3, 1, 0],
    [0, 0, -2, 4, 15, -2, 1],
    [0, 0, 0, 1, -3, 2, 13],
    [0,0,0,0,0,0,1]
])

b = np.array([20, 35, -5, 19, -12, 25,0])


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