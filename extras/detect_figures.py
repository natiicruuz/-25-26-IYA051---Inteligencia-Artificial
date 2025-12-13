"""
Detección de figuras/imágenes en documentos.

Este módulo detecta componentes grandes que no parecen texto y los extrae
como figuras o imágenes independientes.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class FigureDetector:
    """
    Detector de figuras/imágenes en documentos.
    
    Identifica regiones grandes con patrones visuales que no corresponden
    a texto para extraerlas como imágenes independientes.
    """
    
    def __init__(
        self,
        min_area: int = 5000,
        max_aspect_ratio: float = 10.0,
        min_aspect_ratio: float = 0.1
    ):
        """
        Inicializa el detector de figuras.
        
        Args:
            min_area: Área mínima para considerar una región como figura.
            max_aspect_ratio: Relación de aspecto máxima (ancho/alto).
            min_aspect_ratio: Relación de aspecto mínima (ancho/alto).
        """
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio
    
    def detect_figures(
        self,
        image_path: str
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detecta figuras/imágenes en un documento.
        
        Args:
            image_path: Ruta a la imagen del documento.
        
        Returns:
            Lista de tuplas (imagen_figura, bounding_box).
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarización
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operación de cierre para unir componentes cercanos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Encontrar componentes conectados
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        figures = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filtrar por criterios de figura
            if (area > self.min_area and
                self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                
                # Verificar que no sea texto (las figuras suelen tener más variación visual)
                if self._is_likely_figure(gray[y:y+h, x:x+w]):
                    figure_img = image[y:y+h, x:x+w]
                    figures.append((figure_img, (x, y, w, h)))
        
        return figures
    
    def _is_likely_figure(self, region: np.ndarray) -> bool:
        """
        Determina si una región es probablemente una figura y no texto.
        
        Args:
            region: Región de la imagen a analizar.
        
        Returns:
            True si parece una figura, False si parece texto.
        """
        # Calcular varianza de intensidad (figuras suelen tener más variación)
        variance = np.var(region)
        
        # Calcular densidad de píxeles oscuros
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        density = np.sum(binary > 0) / binary.size
        
        # Figuras típicamente tienen:
        # - Mayor varianza que texto simple
        # - Densidad de píxeles en un rango específico
        
        return variance > 1000 and 0.1 < density < 0.7
    
    def visualize_detections(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ):
        """
        Visualiza las figuras detectadas en la imagen.
        
        Args:
            image_path: Ruta a la imagen del documento.
            save_path: Ruta opcional donde guardar la visualización.
        """
        import matplotlib.pyplot as plt
        
        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar figuras
        figures = self.detect_figures(image_path)
        
        # Dibujar bounding boxes
        for idx, (_, (x, y, w, h)) in enumerate(figures):
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(
                image_rgb,
                f"Figura {idx+1}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )
        
        # Visualizar
        plt.figure(figsize=(15, 10))
        plt.imshow(image_rgb)
        plt.title(f'Figuras Detectadas: {len(figures)}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return figures
    
    def save_figures(
        self,
        image_path: str,
        output_dir: str = './extracted_figures'
    ) -> List[str]:
        """
        Detecta y guarda las figuras como imágenes separadas.
        
        Args:
            image_path: Ruta a la imagen del documento.
            output_dir: Directorio donde guardar las figuras extraídas.
        
        Returns:
            Lista de rutas a las figuras guardadas.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detectar figuras
        figures = self.detect_figures(image_path)
        
        saved_paths = []
        
        for idx, (figure_img, bbox) in enumerate(figures):
            output_path = os.path.join(output_dir, f'figure_{idx+1}.png')
            cv2.imwrite(output_path, figure_img)
            saved_paths.append(output_path)
        
        print(f"\n{'='*60}")
        print(f"Figuras Extraídas: {len(figures)}")
        print(f"{'='*60}")
        for i, path in enumerate(saved_paths):
            x, y, w, h = figures[i][1]
            print(f"  Figura {i+1}: {path}")
            print(f"    Dimensiones: {w}x{h} px")
            print(f"    Posición: ({x}, {y})")
        print(f"{'='*60}\n")
        
        return saved_paths
    
    def extract_all_visual_elements(
        self,
        image_path: str,
        output_dir: str = './extracted_elements'
    ) -> dict:
        """
        Extrae todos los elementos visuales (figuras) del documento.
        
        Args:
            image_path: Ruta a la imagen del documento.
            output_dir: Directorio base donde guardar los elementos.
        
        Returns:
            Diccionario con las rutas de los elementos extraídos.
        """
        figures_dir = os.path.join(output_dir, 'figures')
        
        figures = self.save_figures(image_path, figures_dir)
        
        return {
            'figures': figures,
            'total_elements': len(figures)
        }


def main():
    """Función de ejemplo para usar el detector de figuras."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detectar figuras/imágenes en un documento'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen del documento'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./extracted_figures',
        help='Directorio donde guardar las figuras (default: ./extracted_figures)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Mostrar visualización de las figuras detectadas'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=5000,
        help='Área mínima para considerar una figura (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Crear detector
    detector = FigureDetector(min_area=args.min_area)
    
    # Visualizar si se solicita
    if args.visualize:
        detector.visualize_detections(args.image)
    
    # Guardar figuras
    saved_paths = detector.save_figures(args.image, args.output_dir)
    
    return saved_paths


if __name__ == '__main__':
    main()