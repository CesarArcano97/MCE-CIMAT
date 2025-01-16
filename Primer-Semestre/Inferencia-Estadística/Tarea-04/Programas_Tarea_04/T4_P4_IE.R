
#Tarea 4: Problema 4

#a)
#Funcion para estimar la densidad por kernels
estimacion_densidad_kernel <- function(x, h, kernel, datos) {
  n <- length(datos) 
  
  #Función del kernel
  K <- switch(kernel,
              "gaussiano" = function(u) (1 / sqrt(2 * pi)) * exp(-0.5 * u^2),
              "uniforme" = function(u) ifelse(abs(u) <= 1, 0.5, 0),
              "epanechnikov" = function(u) ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0),
              stop("Kernel no soportado"))
  
  #Estimación de densidad
  densidad <- (1 / (n * h)) * sum(K((x - datos) / h))
  
  return(densidad)
}

#b)Cargamos el archivo CSV 
datos <- read.csv("C:/Users/palom/Downloads/Tratamiento.csv")

#Verificamos la estructura del archivo 
str(datos)

#Convertimos a numérico
tratamiento <- as.numeric(na.omit(datos$X1))  

#Verificamos si la columna fue cargada
if (length(tratamiento) == 0) {
  stop("No se encontraron valores numéricos válidos en la columna de tratamiento.")
}

#Creamos una secuencia de puntos para estimar la densidad
x_vals <- seq(min(tratamiento), max(tratamiento), length.out = 100)

#Estimamos la densidad para diferentes valores de h
densidad_h20 <- sapply(x_vals, estimacion_densidad_kernel, h = 20, kernel = "gaussiano", datos = tratamiento)
densidad_h30 <- sapply(x_vals, estimacion_densidad_kernel, h = 30, kernel = "gaussiano", datos = tratamiento)
densidad_h60 <- sapply(x_vals, estimacion_densidad_kernel, h = 60, kernel = "gaussiano", datos = tratamiento)

#Graficamos las densidades 
plot(x_vals, densidad_h20, type = "l", col = "blue", lwd = 2, ylim = c(0, max(densidad_h20, densidad_h30, densidad_h60)),
     xlab = "Duración del Tratamiento (días)", ylab = "Densidad Estimada", main = "Estimación de Densidad con Diferentes h")
lines(x_vals, densidad_h30, col = "red", lwd = 2)
lines(x_vals, densidad_h60, col = "green", lwd = 2)
legend("topright", legend = c("h = 20", "h = 30", "h = 60"), col = c("blue", "red", "green"), lty = 1, lwd = 2)

#c) Función para calcular el ISE
calcular_ISE <- function(h, datos, kernel) {
  n <- length(datos)  
  #Creamos una secuencia de puntos para la estimación de la densidad
  x_vals <- seq(min(datos), max(datos), length.out = 100)
  #Estimamos la densidad para cada valor en x_vals
  densidad_estimada <- sapply(x_vals, estimacion_densidad_kernel, h = h, kernel = kernel, datos = datos)
  
  #Calculamos el ISE como el integral de la densidad al cuadrado
  ISE <- sum(densidad_estimada^2) * (max(x_vals) - min(x_vals)) / length(x_vals)
  
  return(ISE)
}

#Aquí está la función para encontrar el ancho de banda óptimo minimizando el ISE
encontrar_ancho_banda_optimo <- function(datos, kernel = "epanechnikov") {
  #Minimizamos el ISE usando optimización numérica
  optim_result <- optim(par = 1, 
                        fn = calcular_ISE, 
                        datos = datos,
                        kernel = kernel,
                        method = "L-BFGS-B",  
                        lower = 0.01, upper = 100)  
  
  return(optim_result$par)
}

#Cargamos el archivo CSV
datos <- read.csv("C:/Users/palom/Downloads/Tratamiento.csv")

tratamiento <- as.numeric(na.omit(datos$X1))

#Encontramos el ancho de banda óptimo
h_optimo <- encontrar_ancho_banda_optimo(tratamiento, kernel = "epanechnikov")

#Imprimimos el resultado de h
cat("El ancho de banda óptimo es:", h_optimo, "\n")

#Parámetros para graficar la densidad
x_vals <- seq(min(tratamiento), max(tratamiento), length.out = 100)
densidad_optima <- sapply(x_vals, estimacion_densidad_kernel, h = h_optimo, kernel = "epanechnikov", datos = tratamiento)

#Graficamos la densidad estimada con h óptimo
plot(x_vals, densidad_optima, type = "l", col = "blue", lwd = 2, xlab = "Duración del Tratamiento (días)", 
     ylab = "Densidad Estimada", main = paste("Densidad con h óptimo =", round(h_optimo, 2)))




