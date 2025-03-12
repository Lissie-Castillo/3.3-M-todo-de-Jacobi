import numpy as np
import matplotlib.pyplot as plt

 #8x1 +2x2 −x3 = 10
 #3x1 +15x2 −2x3 +x4 = 24
 #−2x2 +12x3 +2x4 −x5 = −18
 #x2 −x3 +9x4 −2x5 +x6 = 16
 #−2x3 +3x4 +14x5 +x6 = −9
 #x4 −2x5 +10x6 = 22

# Definir el sistema de ecuaciones del ejemplo
A = np.array([
    [8, 2, -1, 0, 0, 0],
    [3, 15, -2, 1, 0, 0],
    [0, -2, 12, 2, -1, 0],
    [0, 1, -1, 9, -2, 1],
    [0, 0, -2, 3, 14, 1],
    [0, 0, 0, 1, -2, 10]
])

b = np.array([10, 24, -18, 16, -9, 22])


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