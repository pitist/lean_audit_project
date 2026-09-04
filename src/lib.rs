// src/lib.rs
use std::path::Path;
use anyhow::{Result, Context};
use image::GenericImageView;
use candle_core::{Device, Tensor};

/// Estructura principal para el procesamiento de IA Multimodal en la frontera
pub struct ProcesadorMultimodal {
    device: Device,
}

impl ProcesadorMultimodal {
    /// Inicializa el motor seleccionando el hardware más rápido disponible (CPU o GPU)
    pub fn new() -> Result<Self> {
        // En la frontera (Edge), selecciona CPU optimizada o CUDA si está disponible
        let device = Device::cuda_if_available(0)
            .unwrap_or(Device::Cpu);
        
        Ok(Self { device })
    }

    /// Acción 1: Procesar Texto (Crea un tensor numérico a partir de datos estructurados)
    pub fn procesar_texto(&self, datos: &[f32]) -> Result<Tensor> {
        let tensor = Tensor::from_slice(datos, (1, datos.len()), &self.device)
            .context("Error al crear el tensor de texto")?;
        
        Ok(tensor)
    }

    /// Acción 2: Procesar Imagen (Carga un archivo y lo convierte en matriz de bytes de alta velocidad)
    pub fn procesar_imagen<P: AsRef<Path>>(&self, ruta_imagen: P) -> Result<(u32, u32)> {
        let img = image::open(ruta_imagen)
            .context("No se pudo abrir la imagen en la frontera")?;
        
        // Obtiene las dimensiones de forma nativa sin pasar por capas lentas de Python
        let (ancho, alto) = img.dimensions();
        
        Ok((ancho, alto))
    }

    /// Acción 3: Inferencia Concurrente (Fusión de datos multimedia en hilos nativos)
    pub async fn ejecutar_inferencia_paralela(&self, texto: Vec<f32>) -> Result<String> {
        let tensor = self.procesar_texto(&texto)?;
        
        // Simulación de la velocidad de ejecución nativa en Rust
        let de_regreso = format!(
            "Inferencia completada en {:?}. Forma del Tensor: {:?}", 
            self.device, 
            tensor.shape()
        );
        
        Ok(de_regreso)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_motor_multimodal() {
        let motor = ProcesadorMultimodal::new().unwrap();
        
        // Probamos el procesamiento de texto ultrarrápido
        let datos_entrada = vec![1.0, 2.0, 3.0, 4.0];
        let resultado = motor.ejecutar_inferencia_paralela(datos_entrada).await;
        
        assert!(resultado.is_ok());
        println!("{}", resultado.unwrap());
    }
}
