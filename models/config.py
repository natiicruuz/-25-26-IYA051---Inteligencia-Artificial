"""
Configuración del modelo OCR.

Este módulo define la clase OCRConfig que contiene todos los hiperparámetros,
estructura de entrada y mapeos de caracteres para los diferentes datasets.
"""

from typing import Tuple, List


class OCRConfig:
    """
    Configuración centralizada para el modelo OCR.
    
    Atributos:
        input_shape (tuple): Forma de las imágenes de entrada (canales, alto, ancho).
        num_classes (int): Número de clases a predecir.
        learning_rate (float): Tasa de aprendizaje para el optimizador.
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (1, 28, 28), 
        num_classes: int = 62, 
        learning_rate: float = 0.001
    ):
        """
        Inicializa la configuración del modelo.
        
        Args:
            input_shape: Dimensiones de entrada (canales, altura, ancho).
            num_classes: Número de clases de caracteres (62 para EMNIST, 64 para custom).
            learning_rate: Tasa de aprendizaje inicial.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Mapeo para dataset custom (64 clases: incluye Ñ y ñ)
        self._custom_mapping = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        
        # Mapeo para EMNIST (62 clases: sin Ñ ni ñ)
        self._emnist_mapping = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
    
    def get_character_mapping(self) -> List[str]:
        """
        Retorna el mapeo de caracteres apropiado según el número de clases.
        
        Returns:
            Lista con el mapeo índice -> carácter.
        """
        if self.num_classes == 64:
            return self._custom_mapping
        elif self.num_classes == 62:
            return self._emnist_mapping
        else:
            raise ValueError(f"Número de clases no soportado: {self.num_classes}")
    
    def char_to_index(self, char: str) -> int:
        """
        Convierte un carácter a su índice correspondiente.
        
        Args:
            char: Carácter a convertir.
            
        Returns:
            Índice del carácter en el mapeo.
            
        Raises:
            ValueError: Si el carácter no está en el mapeo.
        """
        mapping = self.get_character_mapping()
        try:
            return mapping.index(char)
        except ValueError:
            raise ValueError(f"Carácter '{char}' no encontrado en el mapeo")
    
    def index_to_char(self, index: int) -> str:
        """
        Convierte un índice a su carácter correspondiente.
        
        Args:
            index: Índice a convertir.
            
        Returns:
            Carácter correspondiente al índice.
            
        Raises:
            IndexError: Si el índice está fuera de rango.
        """
        mapping = self.get_character_mapping()
        if 0 <= index < len(mapping):
            return mapping[index]
        else:
            raise IndexError(f"Índice {index} fuera de rango [0, {len(mapping)-1}]")
    
    def __repr__(self) -> str:
        """Representación en string de la configuración."""
        return (f"OCRConfig(input_shape={self.input_shape}, "
                f"num_classes={self.num_classes}, "
                f"learning_rate={self.learning_rate})")