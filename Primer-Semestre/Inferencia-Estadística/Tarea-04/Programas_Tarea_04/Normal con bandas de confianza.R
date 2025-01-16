#Tenemos los datos del agave
datos_agave <- c(23.37, 21.87, 24.41, 21.27, 23.33, 15.20, 24.21, 25.05, 
                 20.40, 21.05, 28.38, 22.90, 18.00, 17.55, 23.02, 17.32, 
                 30.74, 26.73, 17.22, 22.81, 20.78, 27.52, 15.48, 27.19, 
                 25.92, 23.64, 28.96)


#Graficamos Q-Q plot
grafica_qq_normal(datos_agave)

#c) Para la Q-Q Normal con Bandas de Confianza tenemos que definir lo siguiente
grafica_qq_bandas_confianza <- function(datos, alpha) {
  n <- length(datos)  #Número de datos
  p <- (1:n - 0.5) / n  #Probabilidades
  q <- qnorm(p)  #Quantiles teóricos
  datos_ordenados <- sort(datos)  #Datos ordenados
  
  mu <- mean(datos)  #Media muestral
  sigma <- sd(datos)  #Desviación estándar muestral
  
  #Kolmogorov-Smirnov
  k <- qnorm(1 - alpha / 2) / sqrt(n)  

#Calculamos las bandas de confianza
limite_inferior <- qnorm(pmax(0, p - k)) * sigma + mu  
limite_superior <- qnorm(pmin(1, p + k)) * sigma + mu  

# Para graficar Q-Q plot con bandas de confianza tenemos
plot(q, datos_ordenados, main = paste("Gráfica Q-Q Normal con Bandas de Confianza al", (1 - alpha) * 100, "%"), xlab = "Quantiles Teóricos", ylab = "Quantiles Muestrales")
abline(a = mu, b = sigma, col = "red")  #Línea de referencia
lines(q, limite_inferior, col = "blue", lty = 2)  
lines(q, limite_superior, col = "blue", lty = 2)  
}

#Graficamos Q-Q plot con bandas de confianza 95%
grafica_qq_bandas_confianza(datos_agave, 0.05)

#Graficamos Q-Q plot con bandas de confianza 99%
grafica_qq_bandas_confianza(datos_agave, 0.01)
